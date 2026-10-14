"""
02_analisar_intervalos_teclado.py

Analisa os intervalos temporais do snapshot de teclado do CYBELL-J
antes da reconstrução definitiva das sessões.

Mantém o mesmo critério usado no pipeline de mouse:
- snapshot imutável;
- validação por SHA-256;
- análise por usuária;
- comparação dos limiares 30, 60, 120, 300 e 600 segundos;
- limiar de 300 segundos será usado na próxima etapa para criar session_id.

Este script NÃO cria janelas, features ou modelos.

Saídas:
data/teclado_analise/
    resumo_intervalos_teclado_por_usuario.csv
    sessoes_teclado_por_limiar.csv
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

SNAPSHOT_DIR = BASE_DIR / "data" / "teclado_snapshot"
ANALISE_DIR = BASE_DIR / "data" / "teclado_analise"

CSV_PATH = SNAPSHOT_DIR / "keystrokes_raw.csv"
METADATA_PATH = SNAPSHOT_DIR / "keystrokes_snapshot_metadata.json"

OUT_INTERVALOS = ANALISE_DIR / "resumo_intervalos_teclado_por_usuario.csv"
OUT_LIMIARES = ANALISE_DIR / "sessoes_teclado_por_limiar.csv"

LIMIARES_SEGUNDOS = [30, 60, 120, 300, 600]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)

    return h.hexdigest()


def validar_snapshot() -> dict:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Snapshot de teclado não encontrado:\n{CSV_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadados do snapshot não encontrados:\n{METADATA_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    hash_esperado = metadata.get("sha256")

    if not hash_esperado:
        raise RuntimeError(
            "O arquivo de metadados não contém o SHA-256 esperado."
        )

    hash_atual = sha256_file(CSV_PATH)

    print("\nVerificando integridade do snapshot...")
    print(f"Hash esperado: {hash_esperado}")
    print(f"Hash atual:    {hash_atual}")

    if hash_atual != hash_esperado:
        raise RuntimeError(
            "O snapshot de teclado foi alterado. "
            "A análise foi interrompida."
        )

    print("Snapshot íntegro.")

    return metadata


def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(
        CSV_PATH,
        low_memory=False,
    )

    required = ["user", "timestamp", "event_type"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "Colunas obrigatórias ausentes no snapshot: "
            + ", ".join(missing)
        )

    df = df.copy()

    df["user"] = (
        df["user"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    df["event_type"] = (
        df["event_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    antes = len(df)

    df = df.dropna(
        subset=["user", "timestamp", "event_type"]
    ).copy()

    removidos = antes - len(df)

    if removidos:
        print(
            f"Registros removidos por campos essenciais inválidos: "
            f"{removidos}"
        )

    df = df.sort_values(
        ["user", "timestamp"]
    ).reset_index(drop=True)

    return df


def criar_datetime_local(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert("America/Sao_Paulo")
    )


def main() -> None:
    ANALISE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("CYBELL-J — ANÁLISE DOS INTERVALOS DO TECLADO")
    print("=" * 100)

    metadata = validar_snapshot()

    print("\nCarregando snapshot...")

    df = carregar_dados()

    print(f"Registros válidos: {len(df):,}".replace(",", "."))
    print(f"Usuárias: {df['user'].nunique()}")

    if "_id" in df.columns:
        print(f"IDs duplicados: {int(df['_id'].duplicated().sum())}")

    df["datetime_br"] = criar_datetime_local(df)

    df["gap_segundos"] = (
        df.groupby("user")["timestamp"].diff() / 1000.0
    )

    print("\n" + "=" * 100)
    print("INTERVALOS ENTRE EVENTOS POR USUÁRIA")
    print("=" * 100)

    resumos = []

    for usuario, grupo in df.groupby("user", sort=True):
        gaps = grupo["gap_segundos"].dropna()

        if gaps.empty:
            resumo = {
                "user": usuario,
                "eventos": int(len(grupo)),
                "inicio": "",
                "fim": "",
                "gap_mediana_s": np.nan,
                "gap_p90_s": np.nan,
                "gap_p95_s": np.nan,
                "gap_p99_s": np.nan,
                "gap_max_s": np.nan,
                "gaps_maior_30s": 0,
                "gaps_maior_60s": 0,
                "gaps_maior_120s": 0,
                "gaps_maior_300s": 0,
                "gaps_maior_600s": 0,
            }

            resumos.append(resumo)
            continue

        primeiro = grupo["datetime_br"].iloc[0]
        ultimo = grupo["datetime_br"].iloc[-1]

        resumo = {
            "user": usuario,
            "eventos": int(len(grupo)),
            "inicio": str(primeiro),
            "fim": str(ultimo),
            "gap_mediana_s": float(gaps.median()),
            "gap_p90_s": float(gaps.quantile(0.90)),
            "gap_p95_s": float(gaps.quantile(0.95)),
            "gap_p99_s": float(gaps.quantile(0.99)),
            "gap_max_s": float(gaps.max()),
            "gaps_maior_30s": int((gaps > 30).sum()),
            "gaps_maior_60s": int((gaps > 60).sum()),
            "gaps_maior_120s": int((gaps > 120).sum()),
            "gaps_maior_300s": int((gaps > 300).sum()),
            "gaps_maior_600s": int((gaps > 600).sum()),
        }

        resumos.append(resumo)

        print(f"\nUsuária: {usuario}")
        print(f"  Eventos: {len(grupo)}")
        print(f"  Mediana do intervalo: {gaps.median():.3f} s")
        print(f"  Percentil 90: {gaps.quantile(0.90):.3f} s")
        print(f"  Percentil 95: {gaps.quantile(0.95):.3f} s")
        print(f"  Percentil 99: {gaps.quantile(0.99):.3f} s")
        print(f"  Maior intervalo: {gaps.max():.2f} s")
        print(f"  Pausas > 30 s:  {(gaps > 30).sum()}")
        print(f"  Pausas > 60 s:  {(gaps > 60).sum()}")
        print(f"  Pausas > 120 s: {(gaps > 120).sum()}")
        print(f"  Pausas > 300 s: {(gaps > 300).sum()}")
        print(f"  Pausas > 600 s: {(gaps > 600).sum()}")

    resumo_df = pd.DataFrame(resumos)

    resumo_df.to_csv(
        OUT_INTERVALOS,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("QUANTIDADE DE SESSÕES PARA DIFERENTES LIMIARES")
    print("=" * 100)

    sessoes_rows = []

    for usuario, grupo in df.groupby("user", sort=True):
        print(f"\n{usuario}")

        for limite in LIMIARES_SEGUNDOS:
            nova_sessao = (
                grupo["gap_segundos"].isna()
                | (grupo["gap_segundos"] > limite)
            )

            quantidade = int(nova_sessao.sum())

            sessoes_rows.append(
                {
                    "user": usuario,
                    "limite_segundos": int(limite),
                    "quantidade_sessoes": quantidade,
                }
            )

            print(
                f"  pausa > {limite:>3}s "
                f"-> {quantidade} sessões"
            )

    sessoes_df = pd.DataFrame(sessoes_rows)

    sessoes_df.to_csv(
        OUT_LIMIARES,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("ANÁLISE CONCLUÍDA")
    print("=" * 100)

    print(f"\nSnapshot analisado:")
    print(f"  {CSV_PATH}")

    print(f"\nRegistros esperados no metadata:")
    print(f"  {metadata.get('rows')}")

    print(f"\nArquivos gerados:")
    print(f"  {OUT_INTERVALOS}")
    print(f"  {OUT_LIMIARES}")

    print(
        "\nNenhuma sessão definitiva foi criada ainda. "
        "O próximo passo usará o limiar de 300 segundos, "
        "igual ao pipeline de mouse."
    )


if __name__ == "__main__":
    main()
