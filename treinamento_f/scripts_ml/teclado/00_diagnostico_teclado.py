"""
00_diagnostico_teclado.py

Primeira etapa do pipeline de teclado do CYBELL-J.

Objetivo:
- inspecionar os dados brutos de teclado;
- verificar estrutura, usuários, tipos de evento, timestamps,
  valores ausentes e duplicidades;
- gerar resumos antes de qualquer criação de sessão, janela,
  feature ou treinamento.

Este script NÃO treina modelos e NÃO altera os dados originais.

Fonte de dados:
1. tenta ler a coleção MongoDB "keystrokes", se MONGO_URI estiver disponível;
2. se não houver conexão/configuração, usa data/keystrokes.csv.

Saídas:
data/teclado_analise/
    diagnostico_geral_teclado.csv
    eventos_por_usuario.csv
    eventos_por_tipo.csv
    ausentes_por_coluna.csv
    periodo_por_usuario.csv
    diagnostico_teclado.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "teclado_analise"

CSV_FALLBACK = DATA_DIR / "keystrokes.csv"

MONGO_COLLECTION = "keystrokes"
DEFAULT_DB_NAME = "cybell_db"


def load_from_mongo() -> tuple[pd.DataFrame, str] | None:
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")

    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", DEFAULT_DB_NAME)

    if not uri:
        return None

    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")

        db = client[db_name]

        if MONGO_COLLECTION not in db.list_collection_names():
            client.close()
            return None

        docs = list(db[MONGO_COLLECTION].find())

        for doc in docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        client.close()

        if not docs:
            return pd.DataFrame(), f"MongoDB: {db_name}.{MONGO_COLLECTION}"

        df = pd.json_normalize(docs)
        return df, f"MongoDB: {db_name}.{MONGO_COLLECTION}"

    except Exception as exc:
        print(f"MongoDB indisponível para diagnóstico: {exc}")
        return None


def load_data() -> tuple[pd.DataFrame, str]:
    mongo_result = load_from_mongo()

    if mongo_result is not None:
        return mongo_result

    if CSV_FALLBACK.exists():
        df = pd.read_csv(CSV_FALLBACK, low_memory=False)
        return df, str(CSV_FALLBACK)

    raise FileNotFoundError(
        "Não foi possível localizar os dados de teclado.\n"
        f"- MongoDB: coleção '{MONGO_COLLECTION}' não disponível\n"
        f"- CSV alternativo: {CSV_FALLBACK} não encontrado"
    )


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    columns_lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    return None


def normalize_timestamp_to_ms(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)

    finite = values[np.isfinite(values)]

    if finite.empty:
        return values

    median_abs = float(np.nanmedian(np.abs(finite)))

    if median_abs > 1e17:
        # nanossegundos
        return values / 1_000_000.0

    if median_abs > 1e14:
        # microssegundos
        return values / 1_000.0

    if median_abs > 1e11:
        # milissegundos
        return values

    if median_abs > 1e8:
        # segundos Unix
        return values * 1_000.0

    # timestamps relativos em segundos
    return values * 1_000.0


def normalize_user(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = OUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CYBELL-J — DIAGNÓSTICO INICIAL DO TECLADO")
    print("=" * 100)

    df, source = load_data()

    print(f"\nFonte utilizada: {source}")
    print(f"Registros: {len(df):,}".replace(",", "."))
    print(f"Colunas: {len(df.columns)}")

    if df.empty:
        raise RuntimeError("A fonte de teclado está vazia.")

    user_col = find_column(
        df,
        ["user", "user_id", "username", "data.user", "data.user_id"],
    )
    event_col = find_column(
        df,
        ["event_type", "type", "data.event_type", "data.type"],
    )
    timestamp_col = find_column(
        df,
        [
            "timestamp",
            "data.timestamp",
            "press_time",
            "data.press_time",
            "time",
            "data.time",
        ],
    )
    key_col = find_column(
        df,
        ["key", "data.key", "code", "data.code"],
    )
    hold_col = find_column(
        df,
        ["hold_time", "data.hold_time"],
    )
    id_col = find_column(
        df,
        ["_id", "id"],
    )

    required_missing = []

    if user_col is None:
        required_missing.append("usuário")

    if event_col is None:
        required_missing.append("tipo de evento")

    if timestamp_col is None:
        required_missing.append("timestamp")

    if required_missing:
        raise RuntimeError(
            "Campos essenciais não encontrados: "
            + ", ".join(required_missing)
            + f"\nColunas disponíveis: {list(df.columns)}"
        )

    df = df.copy()

    df["_user"] = normalize_user(df[user_col])
    df["_event_type"] = (
        df[event_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )

    df["_timestamp_numeric"] = pd.to_numeric(
        df[timestamp_col],
        errors="coerce",
    )
    df["_timestamp_ms"] = normalize_timestamp_to_ms(
        df[timestamp_col]
    )

    valid_ts = df["_timestamp_ms"].notna() & np.isfinite(
        df["_timestamp_ms"]
    )

    # Criar a coluna temporal diretamente como timezone-aware (UTC).
    # Isso evita conflito entre datetime tz-naive e tz-aware no pandas.
    df["_datetime_utc"] = pd.to_datetime(
        df["_timestamp_ms"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    duplicate_full_rows = int(df.duplicated().sum())

    duplicate_ids = None

    if id_col is not None:
        duplicate_ids = int(
            df.loc[df[id_col].notna(), id_col].duplicated().sum()
        )

    n_users = int(df["_user"].nunique(dropna=True))
    n_event_types = int(df["_event_type"].nunique(dropna=True))

    first_timestamp = (
        df.loc[valid_ts, "_datetime_utc"].min()
        if valid_ts.any()
        else pd.NaT
    )
    last_timestamp = (
        df.loc[valid_ts, "_datetime_utc"].max()
        if valid_ts.any()
        else pd.NaT
    )

    general_rows = [
        ("fonte", source),
        ("registros_total", int(len(df))),
        ("colunas_total", int(len(df.columns) - 4)),
        ("usuarios_distintos", n_users),
        ("tipos_evento_distintos", n_event_types),
        ("timestamp_invalidos", int((~valid_ts).sum())),
        ("usuarios_ausentes", int(df["_user"].isna().sum())),
        ("tipos_evento_ausentes", int(df["_event_type"].isna().sum())),
        ("linhas_duplicadas_completas", duplicate_full_rows),
        (
            "ids_duplicados",
            duplicate_ids if duplicate_ids is not None else "campo_id_ausente",
        ),
        (
            "inicio_coleta_utc",
            first_timestamp.isoformat()
            if pd.notna(first_timestamp)
            else "",
        ),
        (
            "fim_coleta_utc",
            last_timestamp.isoformat()
            if pd.notna(last_timestamp)
            else "",
        ),
        ("coluna_usuario", user_col),
        ("coluna_event_type", event_col),
        ("coluna_timestamp", timestamp_col),
        ("coluna_tecla", key_col or ""),
        ("coluna_hold_time", hold_col or ""),
    ]

    diagnostico_geral = pd.DataFrame(
        general_rows,
        columns=["metrica", "valor"],
    )

    eventos_usuario = (
        df.dropna(subset=["_user"])
        .groupby("_user", as_index=False)
        .size()
        .rename(
            columns={
                "_user": "user",
                "size": "eventos",
            }
        )
        .sort_values("eventos", ascending=False)
    )

    if len(df) > 0:
        eventos_usuario["percentual_total"] = (
            eventos_usuario["eventos"] / len(df) * 100.0
        )

    eventos_tipo = (
        df.dropna(subset=["_event_type"])
        .groupby("_event_type", as_index=False)
        .size()
        .rename(
            columns={
                "_event_type": "event_type",
                "size": "eventos",
            }
        )
        .sort_values("eventos", ascending=False)
    )

    if len(df) > 0:
        eventos_tipo["percentual_total"] = (
            eventos_tipo["eventos"] / len(df) * 100.0
        )

    ausentes = pd.DataFrame(
        {
            "coluna": df.columns,
            "ausentes": [int(df[c].isna().sum()) for c in df.columns],
        }
    )

    ausentes["percentual_ausente"] = (
        ausentes["ausentes"] / len(df) * 100.0
    )

    ausentes = (
        ausentes[
            ~ausentes["coluna"].astype(str).str.startswith("_")
        ]
        .sort_values(
            ["percentual_ausente", "coluna"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    periodo_usuario_rows = []

    for user, sub in df.dropna(subset=["_user"]).groupby("_user"):
        sub_valid = sub.loc[
            sub["_timestamp_ms"].notna()
            & np.isfinite(sub["_timestamp_ms"])
        ]

        if sub_valid.empty:
            start = ""
            end = ""
            duration_hours = np.nan
        else:
            start_ts = sub_valid["_datetime_utc"].min()
            end_ts = sub_valid["_datetime_utc"].max()

            start = (
                start_ts.isoformat()
                if pd.notna(start_ts)
                else ""
            )
            end = (
                end_ts.isoformat()
                if pd.notna(end_ts)
                else ""
            )

            duration_hours = (
                float(
                    (
                        sub_valid["_timestamp_ms"].max()
                        - sub_valid["_timestamp_ms"].min()
                    )
                    / 3_600_000.0
                )
                if len(sub_valid) > 1
                else 0.0
            )

        periodo_usuario_rows.append(
            {
                "user": user,
                "eventos": int(len(sub)),
                "timestamps_validos": int(len(sub_valid)),
                "inicio_utc": start,
                "fim_utc": end,
                "amplitude_horas": duration_hours,
            }
        )

    periodo_usuario = pd.DataFrame(
        periodo_usuario_rows
    ).sort_values("user")

    files_created = [
        save_table(
            diagnostico_geral,
            "diagnostico_geral_teclado.csv",
        ),
        save_table(
            eventos_usuario,
            "eventos_por_usuario.csv",
        ),
        save_table(
            eventos_tipo,
            "eventos_por_tipo.csv",
        ),
        save_table(
            ausentes,
            "ausentes_por_coluna.csv",
        ),
        save_table(
            periodo_usuario,
            "periodo_por_usuario.csv",
        ),
    ]

    manifest = {
        "source": source,
        "rows": int(len(df)),
        "original_columns": [
            c
            for c in df.columns
            if not str(c).startswith("_")
        ],
        "detected_columns": {
            "user": user_col,
            "event_type": event_col,
            "timestamp": timestamp_col,
            "key": key_col,
            "hold_time": hold_col,
            "id": id_col,
        },
        "users": sorted(
            [
                str(x)
                for x in df["_user"].dropna().unique()
            ]
        ),
        "event_types": sorted(
            [
                str(x)
                for x in df["_event_type"].dropna().unique()
            ]
        ),
        "duplicate_full_rows": duplicate_full_rows,
        "duplicate_ids": duplicate_ids,
        "invalid_timestamps": int((~valid_ts).sum()),
        "next_step": (
            "Analisar intervalos temporais e reconstruir sessões "
            "com gap de 300 segundos."
        ),
    }

    manifest_path = OUT_DIR / "diagnostico_teclado.json"

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False,
        )

    files_created.append(manifest_path)

    print("\nColunas detectadas:")
    print(f"  usuário:       {user_col}")
    print(f"  tipo de evento:{event_col}")
    print(f"  timestamp:     {timestamp_col}")
    print(f"  tecla/código:  {key_col}")
    print(f"  hold_time:     {hold_col}")

    print("\nEventos por usuário:")
    print(eventos_usuario.to_string(index=False))

    print("\nTipos de evento:")
    print(eventos_tipo.to_string(index=False))

    print("\nArquivos gerados:")

    for path in files_created:
        print(f"  {path}")

    print("\nDIAGNÓSTICO CONCLUÍDO.")
    print(
        "Ainda não foram criadas sessões, janelas, features "
        "ou modelos."
    )


if __name__ == "__main__":
    main()
