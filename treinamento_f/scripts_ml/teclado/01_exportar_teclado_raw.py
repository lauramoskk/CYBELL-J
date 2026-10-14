"""
01_exportar_teclado_raw.py

Cria um snapshot imutável da coleção de teclado do CYBELL-J.

Objetivo:
- congelar os dados brutos usados nos experimentos;
- evitar que novas coletas alterem o conjunto durante a modelagem;
- registrar cutoff, contagens e SHA-256 para reprodutibilidade.

Saídas:
data/teclado_snapshot/
    keystrokes_raw.csv
    keystrokes_snapshot_metadata.json

Por segurança, o script não sobrescreve um snapshot existente,
a menos que seja executado com --force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "data" / "teclado_snapshot"
OUT_CSV = OUT_DIR / "keystrokes_raw.csv"
OUT_META = OUT_DIR / "keystrokes_snapshot_metadata.json"

MONGO_COLLECTION = "keystrokes"
DEFAULT_DB_NAME = "cybell_db"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mongo_config() -> tuple[str, str]:
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")

    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", DEFAULT_DB_NAME)

    if not uri:
        raise RuntimeError(
            "MONGO_URI não encontrado. Configure a conexão no arquivo .env "
            "antes de criar o snapshot."
        )

    return uri, db_name


def get_cutoff_id(collection):
    last_doc = collection.find_one(
        {},
        projection={"_id": 1},
        sort=[("_id", -1)],
    )

    if last_doc is None:
        raise RuntimeError("A coleção keystrokes está vazia.")

    return last_doc["_id"]


def normalize_docs(docs: list[dict]) -> pd.DataFrame:
    normalized = []

    for doc in docs:
        item = dict(doc)
        if "_id" in item:
            item["_id"] = str(item["_id"])
        normalized.append(item)

    df = pd.json_normalize(normalized)

    preferred = [
        "_id",
        "user",
        "timestamp",
        "event_type",
        "key",
        "hold_time",
    ]

    ordered = [c for c in preferred if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]

    return df[ordered]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Permite sobrescrever um snapshot existente.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if (OUT_CSV.exists() or OUT_META.exists()) and not args.force:
        raise RuntimeError(
            "Já existe um snapshot de teclado em data/teclado_snapshot.\n"
            "Para preservar a reprodutibilidade, o script foi interrompido.\n"
            "Use --force somente se houver decisão explícita de recriar o snapshot."
        )

    uri, db_name = load_mongo_config()

    from pymongo import MongoClient

    print("=" * 100)
    print("CYBELL-J — SNAPSHOT IMUTÁVEL DO TECLADO")
    print("=" * 100)

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")

    db = client[db_name]

    if MONGO_COLLECTION not in db.list_collection_names():
        client.close()
        raise RuntimeError(
            f"Coleção '{MONGO_COLLECTION}' não encontrada no banco '{db_name}'."
        )

    collection = db[MONGO_COLLECTION]

    cutoff_id = get_cutoff_id(collection)

    print(f"\nBanco: {db_name}")
    print(f"Coleção: {MONGO_COLLECTION}")
    print(f"Cutoff _id: {cutoff_id}")

    query = {"_id": {"$lte": cutoff_id}}
    docs = list(collection.find(query).sort("_id", 1))

    client.close()

    if not docs:
        raise RuntimeError("Nenhum documento encontrado até o cutoff.")

    df = normalize_docs(docs)

    print(f"Registros congelados: {len(df):,}".replace(",", "."))
    print(f"Colunas: {len(df.columns)}")

    df.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8",
    )

    file_hash = sha256_file(OUT_CSV)

    user_counts = {}
    if "user" in df.columns:
        user_counts = {
            str(k): int(v)
            for k, v in (
                df["user"]
                .astype("string")
                .str.strip()
                .value_counts(dropna=False)
                .to_dict()
                .items()
            )
        }

    event_counts = {}
    if "event_type" in df.columns:
        event_counts = {
            str(k): int(v)
            for k, v in (
                df["event_type"]
                .astype("string")
                .str.strip()
                .str.lower()
                .value_counts(dropna=False)
                .to_dict()
                .items()
            )
        }

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": db_name,
        "collection": MONGO_COLLECTION,
        "cutoff_id": str(cutoff_id),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "users": user_counts,
        "event_types": event_counts,
        "csv_path": str(OUT_CSV.relative_to(BASE_DIR)),
        "sha256": file_hash,
        "important": (
            "Este arquivo é a fonte congelada para os experimentos de teclado. "
            "Não reexportar durante a mesma rodada experimental."
        ),
    }

    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nEventos por usuário:")
    if user_counts:
        for user, count in user_counts.items():
            print(f"  {user}: {count:,}".replace(",", "."))
    else:
        print("  coluna 'user' não encontrada")

    print("\nTipos de evento:")
    if event_counts:
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count:,}".replace(",", "."))
    else:
        print("  coluna 'event_type' não encontrada")

    print("\nSnapshot salvo:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_META}")
    print(f"\nSHA-256:")
    print(f"  {file_hash}")

    print("\nSNAPSHOT CONCLUÍDO.")
    print(
        "A partir da próxima etapa, os scripts de teclado devem ler "
        "keystrokes_raw.csv, e não consultar o MongoDB."
    )


if __name__ == "__main__":
    main()