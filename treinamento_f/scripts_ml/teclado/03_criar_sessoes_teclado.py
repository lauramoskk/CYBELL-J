"""
03_criar_sessoes_teclado.py

Reconstrói as sessões do teclado do CYBELL-J usando o mesmo critério
temporal adotado no pipeline de mouse:

    nova sessão quando o intervalo entre eventos > 300 segundos.

Este script:
- valida a integridade do snapshot imutável;
- ordena os eventos por usuária e timestamp;
- cria session_id;
- gera um CSV de eventos com sessão;
- gera um resumo por sessão;
- registra metadados do processamento.

Este script NÃO cria janelas, features ou modelos.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

SNAPSHOT_DIR = BASE_DIR / "data" / "teclado_snapshot"
ANALISE_DIR = BASE_DIR / "data" / "teclado_analise"
DATASET_DIR = BASE_DIR / "data" / "teclado_dataset"

RAW_CSV = SNAPSHOT_DIR / "keystrokes_raw.csv"
RAW_META = SNAPSHOT_DIR / "keystrokes_snapshot_metadata.json"

OUT_EVENTS = DATASET_DIR / "keystrokes_com_sessoes.csv"
OUT_SESSIONS = ANALISE_DIR / "resumo_sessoes_teclado_300s.csv"
OUT_META = DATASET_DIR / "sessoes_teclado_metadata.json"

SESSION_GAP_SECONDS = 300


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validar_snapshot() -> dict:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Snapshot não encontrado:\n{RAW_CSV}")

    if not RAW_META.exists():
        raise FileNotFoundError(f"Metadados não encontrados:\n{RAW_META}")

    with RAW_META.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    esperado = meta.get("sha256")
    atual = sha256_file(RAW_CSV)

    print("\nVerificando integridade do snapshot...")
    print(f"Hash esperado: {esperado}")
    print(f"Hash atual:    {atual}")

    if not esperado or atual != esperado:
        raise RuntimeError(
            "O snapshot de teclado não corresponde ao arquivo congelado."
        )

    print("Snapshot íntegro.")
    return meta


def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, low_memory=False)

    obrigatorias = ["user", "timestamp", "event_type"]
    faltantes = [c for c in obrigatorias if c not in df.columns]

    if faltantes:
        raise RuntimeError(
            "Colunas obrigatórias ausentes: " + ", ".join(faltantes)
        )

    df = df.copy()

    df["user"] = (
        df["user"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df["event_type"] = (
        df["event_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )

    antes = len(df)

    df = df.dropna(
        subset=["user", "timestamp", "event_type"]
    ).copy()

    removidos = antes - len(df)
    if removidos:
        print(f"Registros inválidos removidos: {removidos}")

    ordem = ["user", "timestamp"]
    if "_id" in df.columns:
        ordem.append("_id")

    df = df.sort_values(ordem).reset_index(drop=True)

    return df


def criar_sessoes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gap_segundos"] = (
        df.groupby("user")["timestamp"].diff() / 1000.0
    )

    df["_nova_sessao"] = (
        df["gap_segundos"].isna()
        | (df["gap_segundos"] > SESSION_GAP_SECONDS)
    )

    df["_session_number"] = (
        df.groupby("user")["_nova_sessao"].cumsum().astype(int)
    )

    df["session_id"] = (
        df["user"].astype(str)
        + "_S"
        + df["_session_number"].astype(str).str.zfill(2)
    )

    # gap_segundos mede a pausa em relação ao evento anterior da usuária.
    # No primeiro evento de uma nova sessão, esse valor é justamente o gap
    # que causou a quebra (>300 s). Para validar gaps DENTRO da sessão,
    # precisamos zerar/ignorar esse primeiro intervalo.
    df["gap_interno_segundos"] = df["gap_segundos"].where(
        ~df["_nova_sessao"],
        np.nan,
    )

    df["datetime_br"] = (
        pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert("America/Sao_Paulo")
        .astype("string")
    )

    return df


def gerar_resumo_sessoes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (user, session_id), g in df.groupby(
        ["user", "session_id"],
        sort=True,
    ):
        start_ms = float(g["timestamp"].min())
        end_ms = float(g["timestamp"].max())

        duracao_s = max(0.0, (end_ms - start_ms) / 1000.0)

        counts = g["event_type"].value_counts().to_dict()

        rows.append(
            {
                "user": user,
                "session_id": session_id,
                "eventos": int(len(g)),
                "keydown": int(counts.get("keydown", 0)),
                "shortcut": int(counts.get("shortcut", 0)),
                "inicio_timestamp_ms": int(start_ms),
                "fim_timestamp_ms": int(end_ms),
                "duracao_segundos": duracao_s,
                "duracao_minutos": duracao_s / 60.0,
                "inicio_br": g["datetime_br"].iloc[0],
                "fim_br": g["datetime_br"].iloc[-1],
                "maior_gap_interno_s": float(
                    g["gap_interno_segundos"].dropna().max()
                    if not g["gap_interno_segundos"].dropna().empty
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def validar_resultado(
    eventos: pd.DataFrame,
    sessoes: pd.DataFrame,
) -> None:
    if eventos["session_id"].isna().any():
        raise RuntimeError("Há eventos sem session_id.")

    if eventos["user"].isna().any():
        raise RuntimeError("Há eventos sem usuária.")

    if eventos["timestamp"].isna().any():
        raise RuntimeError("Há eventos sem timestamp válido.")

    # Dentro de cada sessão, nenhum intervalo pode ser maior que 300 s.
    # O gap que iniciou a sessão é excluído por definição.
    maiores = (
        eventos.groupby(["user", "session_id"])["gap_interno_segundos"]
        .max()
        .dropna()
    )

    if (maiores > SESSION_GAP_SECONDS).any():
        problematicos = maiores[maiores > SESSION_GAP_SECONDS]
        raise RuntimeError(
            "Foram encontrados gaps > 300 s dentro de sessões:\n"
            + str(problematicos)
        )

    # Quantidade de sessões deve ser coerente entre eventos e resumo.
    a = eventos.groupby("user")["session_id"].nunique().sort_index()
    b = sessoes.groupby("user")["session_id"].nunique().sort_index()

    if not a.equals(b):
        raise RuntimeError(
            "Inconsistência entre as sessões dos eventos e o resumo."
        )


def main() -> None:
    ANALISE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CYBELL-J — CRIAÇÃO DAS SESSÕES DO TECLADO")
    print("=" * 100)
    print(f"\nCritério: nova sessão após pausa > {SESSION_GAP_SECONDS} segundos.")

    snapshot_meta = validar_snapshot()

    print("\nCarregando snapshot...")
    df = carregar_dados()

    print(f"Eventos válidos: {len(df):,}".replace(",", "."))
    print(f"Usuárias: {df['user'].nunique()}")

    eventos = criar_sessoes(df)
    sessoes = gerar_resumo_sessoes(eventos)

    validar_resultado(eventos, sessoes)

    # Remover colunas auxiliares antes de salvar.
    colunas_auxiliares = [
        "_nova_sessao",
        "_session_number",
    ]

    eventos_saida = eventos.drop(
        columns=[c for c in colunas_auxiliares if c in eventos.columns]
    )

    eventos_saida.to_csv(
        OUT_EVENTS,
        index=False,
        encoding="utf-8",
    )

    sessoes.to_csv(
        OUT_SESSIONS,
        index=False,
        encoding="utf-8-sig",
    )

    sessoes_por_user = (
        sessoes.groupby("user")
        .agg(
            sessoes=("session_id", "nunique"),
            eventos=("eventos", "sum"),
            eventos_mediana_sessao=("eventos", "median"),
            duracao_mediana_min=("duracao_minutos", "median"),
            duracao_total_min=("duracao_minutos", "sum"),
        )
        .reset_index()
        .sort_values("user")
    )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(RAW_CSV.relative_to(BASE_DIR)),
        "source_sha256": snapshot_meta.get("sha256"),
        "session_gap_seconds": SESSION_GAP_SECONDS,
        "rows_input": int(len(df)),
        "rows_output": int(len(eventos_saida)),
        "users": int(eventos_saida["user"].nunique()),
        "sessions_total": int(sessoes["session_id"].nunique()),
        "sessions_by_user": {
            str(row["user"]): int(row["sessoes"])
            for _, row in sessoes_por_user.iterrows()
        },
        "output_events": str(OUT_EVENTS.relative_to(BASE_DIR)),
        "output_sessions": str(OUT_SESSIONS.relative_to(BASE_DIR)),
        "next_step": (
            "Gerar janelas de teclado sem atravessar session_id e "
            "extrair features específicas de teclado."
        ),
    }

    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nSessões criadas por usuária:")
    print(sessoes_por_user.to_string(index=False))

    print(f"\nTotal de sessões: {sessoes['session_id'].nunique()}")

    print("\nArquivos gerados:")
    print(f"  {OUT_EVENTS}")
    print(f"  {OUT_SESSIONS}")
    print(f"  {OUT_META}")

    print("\nVALIDAÇÕES:")
    print("  Todos os eventos possuem session_id.")
    print("  Nenhuma sessão contém gap interno > 300 s.")
    print("  Contagem de sessões consistente.")

    print("\nSESSÕES DO TECLADO CONCLUÍDAS.")
    print(
        "Ainda não foram criadas janelas, features ou modelos."
    )


if __name__ == "__main__":
    main()
