"""
00_auditoria_trackpad.py

CYBELL-J — Auditoria inicial do trackpad

Objetivo:
- analisar SOMENTE janelas classificadas como trackpad com alta confiança;
- não misturar mouse;
- verificar quantidade de dados, sessões e configurações disponíveis;
- avaliar se há base suficiente para um experimento de reconhecimento
  específico de trackpad.

Fonte:
data/mouse_dataset/dataset_device_high_confidence.csv

O arquivo de origem já foi produzido pelo pipeline de mouse/ponteiro e contém
as mesmas features, janelas e critérios de sessão. Aqui apenas auditamos a
parcela TRACKPAD, sem treinar nenhum modelo.

Saídas:
data/trackpad_analise/
    trackpad_por_usuario.csv
    trackpad_por_configuracao.csv
    trackpad_por_usuario_configuracao.csv
    trackpad_por_sessao.csv
    auditoria_trackpad.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

SOURCE_DIR = BASE_DIR / "data" / "mouse_dataset"
SOURCE_CSV = SOURCE_DIR / "dataset_device_high_confidence.csv"
SOURCE_MANIFEST = SOURCE_DIR / "dataset_manifest.json"

OUT_DIR = BASE_DIR / "data" / "trackpad_analise"

OUT_USER = OUT_DIR / "trackpad_por_usuario.csv"
OUT_CONFIG = OUT_DIR / "trackpad_por_configuracao.csv"
OUT_USER_CONFIG = OUT_DIR / "trackpad_por_usuario_configuracao.csv"
OUT_SESSION = OUT_DIR / "trackpad_por_sessao.csv"
OUT_JSON = OUT_DIR / "auditoria_trackpad.json"

EXPECTED_WINDOWS = [25, 30, 50, 100]
EXPECTED_OVERLAPS = [25, 50, 75]
MIN_FOLDS = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def high_confidence_mask(series: pd.Series) -> pd.Series:
    """
    Aceita bool, 0/1 e strings comuns.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.eq(1)

    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "sim"])
    )

    return numeric_mask | text


def validar_fonte() -> dict:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Dataset de alta confiança não encontrado:\n{SOURCE_CSV}"
        )

    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifesto do dataset não encontrado:\n{SOURCE_MANIFEST}"
        )

    with SOURCE_MANIFEST.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_hash = manifest.get("dataset_device_sha256")
    current_hash = sha256_file(SOURCE_CSV)

    print("\nVerificando integridade do dataset de origem...")
    print(f"Hash esperado: {expected_hash}")
    print(f"Hash atual:    {current_hash}")

    if expected_hash and current_hash != expected_hash:
        raise RuntimeError(
            "O hash do dataset_device_high_confidence.csv não corresponde "
            "ao manifesto. Não continuar até verificar a origem."
        )

    print("Dataset de origem íntegro.")
    return manifest


def carregar_trackpad() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(SOURCE_CSV, low_memory=False)

    required = [
        "user",
        "session_id",
        "window_id",
        "window_size",
        "requested_overlap_pct",
        "device_context",
        "device_high_confidence",
        "device_trackpad_pct",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Colunas obrigatórias ausentes: " + ", ".join(missing)
        )

    df = df.copy()

    df["user"] = df["user"].astype("string").str.strip()
    df["session_id"] = df["session_id"].astype("string").str.strip()
    df["device_context"] = (
        df["device_context"].astype("string").str.strip().str.lower()
    )

    df["window_size"] = pd.to_numeric(
        df["window_size"], errors="coerce"
    )

    df["requested_overlap_pct"] = pd.to_numeric(
        df["requested_overlap_pct"], errors="coerce"
    )

    df["device_trackpad_pct"] = pd.to_numeric(
        df["device_trackpad_pct"], errors="coerce"
    )

    hc = high_confidence_mask(df["device_high_confidence"])

    trackpad = df[
        df["device_context"].eq("trackpad") & hc
    ].copy()

    trackpad = trackpad.dropna(
        subset=[
            "user",
            "session_id",
            "window_id",
            "window_size",
            "requested_overlap_pct",
        ]
    ).copy()

    trackpad["window_size"] = trackpad["window_size"].astype(int)
    trackpad["requested_overlap_pct"] = (
        trackpad["requested_overlap_pct"].astype(int)
    )

    return df, trackpad


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CYBELL-J — AUDITORIA INICIAL DO TRACKPAD")
    print("=" * 100)
    print("\nRegra desta etapa:")
    print("  analisar apenas TRACKPAD de alta confiança")
    print("  nenhum dado classificado como mouse entra nesta análise")
    print("  nenhum modelo será treinado")

    manifest = validar_fonte()

    print("\nCarregando dataset...")
    full, trackpad = carregar_trackpad()

    print(f"Janelas totais na fonte: {len(full):,}".replace(",", "."))
    print(
        f"Janelas trackpad de alta confiança: "
        f"{len(trackpad):,}".replace(",", ".")
    )

    if trackpad.empty:
        raise RuntimeError(
            "Nenhuma janela de trackpad de alta confiança foi encontrada."
        )

    print(f"Usuárias com trackpad: {trackpad['user'].nunique()}")
    print(f"Sessões com trackpad: {trackpad['session_id'].nunique()}")

    by_user = (
        trackpad.groupby("user")
        .agg(
            janelas=("window_id", "size"),
            sessoes=("session_id", "nunique"),
            trackpad_pct_media=("device_trackpad_pct", "mean"),
            trackpad_pct_mediana=("device_trackpad_pct", "median"),
            trackpad_pct_min=("device_trackpad_pct", "min"),
        )
        .reset_index()
        .sort_values(
            ["janelas", "sessoes"],
            ascending=[False, False],
        )
    )

    by_user["apto_3_folds_por_sessao"] = (
        by_user["sessoes"] >= MIN_FOLDS
    )

    by_config = (
        trackpad.groupby(
            ["window_size", "requested_overlap_pct"]
        )
        .agg(
            janelas=("window_id", "size"),
            usuarios=("user", "nunique"),
            sessoes=("session_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["window_size", "requested_overlap_pct"]
        )
    )

    by_user_config = (
        trackpad.groupby(
            [
                "user",
                "window_size",
                "requested_overlap_pct",
            ]
        )
        .agg(
            janelas=("window_id", "size"),
            sessoes=("session_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            [
                "user",
                "window_size",
                "requested_overlap_pct",
            ]
        )
    )

    by_session = (
        trackpad.groupby(
            ["user", "session_id"]
        )
        .agg(
            janelas=("window_id", "size"),
            configuracoes=(
                "window_size",
                lambda s: int(
                    trackpad.loc[s.index, [
                        "window_size",
                        "requested_overlap_pct",
                    ]]
                    .drop_duplicates()
                    .shape[0]
                ),
            ),
            trackpad_pct_media=("device_trackpad_pct", "mean"),
            trackpad_pct_min=("device_trackpad_pct", "min"),
        )
        .reset_index()
        .sort_values(["user", "session_id"])
    )

    by_user.to_csv(
        OUT_USER,
        index=False,
        encoding="utf-8-sig",
    )

    by_config.to_csv(
        OUT_CONFIG,
        index=False,
        encoding="utf-8-sig",
    )

    by_user_config.to_csv(
        OUT_USER_CONFIG,
        index=False,
        encoding="utf-8-sig",
    )

    by_session.to_csv(
        OUT_SESSION,
        index=False,
        encoding="utf-8-sig",
    )

    users_with_3_sessions = (
        by_user.loc[
            by_user["sessoes"] >= MIN_FOLDS,
            "user",
        ]
        .astype(str)
        .tolist()
    )

    # Para reconhecimento entre pessoas, não basta uma pessoa ter 3 sessões.
    # Precisamos de pelo menos duas pessoas com dados independentes suficientes.
    recognition_feasible = len(users_with_3_sessions) >= 2

    expected_configs = {
        (w, o)
        for w in EXPECTED_WINDOWS
        for o in EXPECTED_OVERLAPS
    }

    observed_configs = {
        (int(r.window_size), int(r.requested_overlap_pct))
        for r in by_config.itertuples()
    }

    missing_configs = sorted(
        expected_configs - observed_configs
    )

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE_CSV.relative_to(BASE_DIR)),
        "source_sha256": manifest.get("dataset_device_sha256"),
        "selection": {
            "device_context": "trackpad",
            "device_high_confidence": True,
            "mouse_included": False,
        },
        "session_gap_seconds_inherited": manifest.get(
            "session_gap_seconds"
        ),
        "window_sizes_expected": EXPECTED_WINDOWS,
        "overlaps_expected_pct": EXPECTED_OVERLAPS,
        "windows_trackpad_high_confidence": int(len(trackpad)),
        "users_with_trackpad": int(trackpad["user"].nunique()),
        "sessions_with_trackpad": int(trackpad["session_id"].nunique()),
        "users_with_at_least_3_trackpad_sessions": users_with_3_sessions,
        "recognition_training_feasible_with_3_session_folds": (
            recognition_feasible
        ),
        "missing_window_overlap_configs": [
            {
                "window_size": int(w),
                "overlap_pct": int(o),
            }
            for w, o in missing_configs
        ],
        "normalization_policy_if_training_is_possible": {
            "same_as_mouse": True,
            "imputation": "SimpleImputer(strategy='median')",
            "scaling": "none for RandomForest/XGBoost",
            "split": "3 folds by session_id",
            "random_state": 42,
            "note": (
                "Trackpad usa o mesmo protocolo de pré-processamento e "
                "validação do mouse; não se misturam eventos de mouse."
            ),
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            audit,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print("TRACKPAD POR USUÁRIA")
    print("=" * 100)
    print(by_user.to_string(index=False))

    print("\n" + "=" * 100)
    print("TRACKPAD POR CONFIGURAÇÃO")
    print("=" * 100)
    print(by_config.to_string(index=False))

    print("\n" + "=" * 100)
    print("AVALIAÇÃO PARA TREINAMENTO DE RECONHECIMENTO")
    print("=" * 100)

    print(
        "Usuárias com pelo menos 3 sessões independentes de trackpad: "
        + (
            ", ".join(users_with_3_sessions)
            if users_with_3_sessions
            else "nenhuma"
        )
    )

    if recognition_feasible:
        print(
            "Há pelo menos duas usuárias com 3+ sessões. "
            "É possível avançar para um treinamento específico de trackpad."
        )
    else:
        print(
            "Não há pelo menos duas usuárias com 3+ sessões independentes "
            "de trackpad."
        )
        print(
            "Portanto, um experimento de reconhecimento entre pessoas com "
            "3-fold por sessão NÃO é metodologicamente sustentado nesta base."
        )
        print(
            "Não completar a base com mouse: isso violaria o Experimento 1.2."
        )

    print("\nArquivos gerados:")
    print(f"  {OUT_USER}")
    print(f"  {OUT_CONFIG}")
    print(f"  {OUT_USER_CONFIG}")
    print(f"  {OUT_SESSION}")
    print(f"  {OUT_JSON}")

    print("\nAUDITORIA DO TRACKPAD CONCLUÍDA.")
    print("Nenhum modelo foi treinado.")


if __name__ == "__main__":
    main()
