"""
06_documentar_experimento1_1_mouse.py

CYBELL-J — Experimento 1.1
ANÁLISE FINAL DO MOUSE, SEM TRACKPAD E SEM NOVO TREINAMENTO

Objetivo:
- separar fisicamente a base de mouse de alta confiança;
- comprovar que nenhum trackpad entra na base do mouse;
- analisar os dados de entrada;
- documentar quantidade de usuárias, sessões, janelas e configurações;
- registrar o mesmo protocolo de pré-processamento/validação usado nos modelos.

Fonte:
data/mouse_dataset/dataset_device_high_confidence.csv

Saídas:
data/mouse_experimento1_1/
    dataset_mouse_high_confidence.csv
    resumo_mouse_final.csv
    sessoes_mouse_final.csv
    mouse_por_configuracao.csv
    conclusao_experimento1_1.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

SOURCE_CSV = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)

SOURCE_MANIFEST = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_manifest.json"
)

OUT_DIR = (
    BASE_DIR
    / "data"
    / "mouse_experimento1_1"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DATASET = OUT_DIR / "dataset_mouse_high_confidence.csv"
OUT_SUMMARY = OUT_DIR / "resumo_mouse_final.csv"
OUT_SESSIONS = OUT_DIR / "sessoes_mouse_final.csv"
OUT_CONFIG = OUT_DIR / "mouse_por_configuracao.csv"
OUT_CONCLUSION = OUT_DIR / "conclusao_experimento1_1.json"

EXPECTED_WINDOWS = [25, 30, 50, 100]
EXPECTED_OVERLAPS = [25, 50, 75]
MIN_FOLDS = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado:\n{SOURCE_MANIFEST}"
        )

    with SOURCE_MANIFEST.open("r", encoding="utf-8") as f:
        return json.load(f)


def high_confidence_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return numeric.eq(1) | text.isin(["true", "1", "yes", "sim"])


def main() -> None:
    print("=" * 100)
    print("CYBELL-J — EXPERIMENTO 1.1 — ANÁLISE FINAL DO MOUSE")
    print("=" * 100)

    print("\nRegras:")
    print("  Somente mouse de alta confiança.")
    print("  Nenhum dado de trackpad será incluído.")
    print("  Nenhum novo modelo será treinado nesta etapa.")

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Base de origem não encontrada:\n{SOURCE_CSV}"
        )

    manifest = load_manifest()

    expected_hash = manifest.get("dataset_device_sha256")
    current_hash = sha256_file(SOURCE_CSV)

    print("\nVerificando integridade da fonte...")
    print(f"Hash esperado: {expected_hash}")
    print(f"Hash atual:    {current_hash}")

    if expected_hash and current_hash != expected_hash:
        raise RuntimeError(
            "Hash da base de origem divergente."
        )

    print("Fonte íntegra.")

    df = pd.read_csv(
        SOURCE_CSV,
        low_memory=False,
    )

    required = [
        "user",
        "session_id",
        "window_id",
        "window_size",
        "requested_overlap_pct",
        "device_context",
        "device_high_confidence",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(missing)
        )

    df = df.copy()

    df["user"] = (
        df["user"]
        .astype("string")
        .str.strip()
    )

    df["session_id"] = (
        df["session_id"]
        .astype("string")
        .str.strip()
    )

    df["device_context"] = (
        df["device_context"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["window_size"] = pd.to_numeric(
        df["window_size"],
        errors="coerce",
    )

    df["requested_overlap_pct"] = pd.to_numeric(
        df["requested_overlap_pct"],
        errors="coerce",
    )

    hc = high_confidence_mask(
        df["device_high_confidence"]
    )

    mouse = df[
        df["device_context"].eq("mouse")
        & hc
    ].copy()

    mouse = mouse.dropna(
        subset=[
            "user",
            "session_id",
            "window_id",
            "window_size",
            "requested_overlap_pct",
        ]
    ).reset_index(drop=True)

    mouse["window_size"] = (
        mouse["window_size"].astype(int)
    )

    mouse["requested_overlap_pct"] = (
        mouse["requested_overlap_pct"].astype(int)
    )

    if mouse.empty:
        raise RuntimeError(
            "Nenhuma janela de mouse de alta confiança encontrada."
        )

    if not mouse["device_context"].eq("mouse").all():
        raise RuntimeError(
            "Foi encontrado dado não-mouse na base filtrada."
        )

    if (
        mouse["device_context"]
        .astype(str)
        .str.lower()
        .eq("trackpad")
        .any()
    ):
        raise RuntimeError(
            "Trackpad detectado na base do mouse."
        )

    mouse.to_csv(
        OUT_DATASET,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nJanelas mouse de alta confiança: "
        f"{len(mouse):,}".replace(",", ".")
    )
    print(
        f"Usuárias: {mouse['user'].nunique()}"
    )
    print(
        f"Sessões representadas: "
        f"{mouse['session_id'].nunique()}"
    )

    summary = (
        mouse.groupby("user")
        .agg(
            janelas=("window_id", "size"),
            sessoes=("session_id", "nunique"),
        )
        .reset_index()
        .sort_values("user")
    )

    summary["apto_3_folds"] = (
        summary["sessoes"] >= MIN_FOLDS
    )

    if "device_mouse_pct" in mouse.columns:
        mouse["device_mouse_pct"] = pd.to_numeric(
            mouse["device_mouse_pct"],
            errors="coerce",
        )

        quality = (
            mouse.groupby("user")["device_mouse_pct"]
            .agg(["mean", "median", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "mean": "mouse_pct_mean",
                    "median": "mouse_pct_median",
                    "min": "mouse_pct_min",
                    "max": "mouse_pct_max",
                }
            )
        )

        summary = summary.merge(
            quality,
            on="user",
            how="left",
        )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    sessions = (
        mouse.groupby(
            ["user", "session_id"]
        )
        .agg(
            janelas=("window_id", "size"),
            menor_janela=("window_size", "min"),
            maior_janela=("window_size", "max"),
        )
        .reset_index()
        .sort_values(
            ["user", "session_id"]
        )
    )

    sessions.to_csv(
        OUT_SESSIONS,
        index=False,
        encoding="utf-8-sig",
    )

    by_config = (
        mouse.groupby(
            [
                "window_size",
                "requested_overlap_pct",
            ]
        )
        .agg(
            janelas=("window_id", "size"),
            usuarios=("user", "nunique"),
            sessoes=("session_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            [
                "window_size",
                "requested_overlap_pct",
            ]
        )
    )

    by_config.to_csv(
        OUT_CONFIG,
        index=False,
        encoding="utf-8-sig",
    )

    expected_configs = {
        (w, o)
        for w in EXPECTED_WINDOWS
        for o in EXPECTED_OVERLAPS
    }

    observed_configs = {
        (
            int(row.window_size),
            int(row.requested_overlap_pct),
        )
        for row in by_config.itertuples()
    }

    users_3folds = (
        summary.loc[
            summary["sessoes"] >= MIN_FOLDS,
            "user",
        ]
        .astype(str)
        .tolist()
    )

    all_users_ready = (
        len(users_3folds)
        == mouse["user"].nunique()
    )

    conclusion = {
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "experiment": (
            "Experimento 1.1 - Mouse sem trackpad"
        ),
        "source": str(
            SOURCE_CSV.relative_to(BASE_DIR)
        ),
        "source_sha256": current_hash,
        "selection": {
            "device_context": "mouse",
            "high_confidence_only": True,
            "trackpad_included": False,
        },
        "input_analysis": {
            "windows": int(len(mouse)),
            "users": int(
                mouse["user"].nunique()
            ),
            "sessions": int(
                mouse["session_id"].nunique()
            ),
            "observed_configurations": int(
                len(observed_configs)
            ),
            "missing_configurations": [
                {
                    "window_size": int(w),
                    "overlap_pct": int(o),
                }
                for w, o in sorted(
                    expected_configs
                    - observed_configs
                )
            ],
        },
        "users_with_at_least_3_sessions": (
            users_3folds
        ),
        "all_users_ready_for_3fold_session_cv": (
            all_users_ready
        ),
        "training_protocol": {
            "same_preprocessing_across_mouse_keyboard_trackpad": True,
            "imputation": (
                "SimpleImputer(strategy='median')"
            ),
            "scaling": (
                "none for RandomForest/XGBoost"
            ),
            "cross_validation": (
                "3 folds by session_id"
            ),
            "random_state": 42,
            "mouse_only": True,
            "trackpad_included": False,
        },
        "models_already_used": {
            "experiment_2_global": (
                "scripts_ml/mouse/"
                "10A_identificacao_mouse_6_usuarios.py"
            ),
            "experiment_1_individual": (
                "scripts_ml/mouse/"
                "10B_autenticacao_continua_mouse.py"
            ),
        },
        "status": (
            "Experimento 1.1 documentado: "
            "base de mouse separada e sem trackpad."
        ),
    }

    with OUT_CONCLUSION.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            conclusion,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print("RESUMO FINAL DO MOUSE")
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nConfigurações janela/overlap presentes: "
        f"{len(observed_configs)}/12"
    )

    print(
        "\nUsuárias com pelo menos "
        "3 sessões independentes:"
    )
    print(
        "  " + ", ".join(users_3folds)
    )

    print("\nValidação Experimento 1.1:")
    print("  Base contém somente mouse de alta confiança.")
    print("  Trackpad incluído: NÃO.")
    print(
        "  Base compatível com 3-fold por sessão: "
        + ("SIM" if all_users_ready else "NÃO para todas")
    )

    print("\nArquivos gerados:")
    print(f"  {OUT_DATASET}")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_SESSIONS}")
    print(f"  {OUT_CONFIG}")
    print(f"  {OUT_CONCLUSION}")

    print("\nEXPERIMENTO 1.1 DOCUMENTADO.")


if __name__ == "__main__":
    main()
