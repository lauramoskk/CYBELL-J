"""
01_analisar_trackpad.py

CYBELL-J — Experimento 1.2
ANÁLISE DESCRITIVA DO TRACKPAD, SEM MOUSE E SEM TREINAMENTO

Objetivo:
- separar fisicamente a base de trackpad de alta confiança;
- analisar os dados de entrada;
- manter o mesmo padrão de sessões/janelas usado no mouse;
- documentar a insuficiência amostral para reconhecimento entre pessoas;
- NÃO treinar modelos e NÃO completar a base com dados de mouse.

Fonte:
data/mouse_dataset/dataset_device_high_confidence.csv

Saídas:
data/trackpad_dataset/
    dataset_trackpad_high_confidence.csv
    dataset_trackpad_manifest.json

data/trackpad_analise/
    resumo_trackpad_final.csv
    sessoes_trackpad_final.csv
    qualidade_features_trackpad.csv
    conclusao_experimento1_2.json
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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

TRACKPAD_DATASET_DIR = BASE_DIR / "data" / "trackpad_dataset"
TRACKPAD_ANALYSIS_DIR = BASE_DIR / "data" / "trackpad_analise"

OUT_DATASET = (
    TRACKPAD_DATASET_DIR
    / "dataset_trackpad_high_confidence.csv"
)

OUT_DATASET_MANIFEST = (
    TRACKPAD_DATASET_DIR
    / "dataset_trackpad_manifest.json"
)

OUT_SUMMARY = (
    TRACKPAD_ANALYSIS_DIR
    / "resumo_trackpad_final.csv"
)

OUT_SESSIONS = (
    TRACKPAD_ANALYSIS_DIR
    / "sessoes_trackpad_final.csv"
)

OUT_FEATURE_QUALITY = (
    TRACKPAD_ANALYSIS_DIR
    / "qualidade_features_trackpad.csv"
)

OUT_CONCLUSION = (
    TRACKPAD_ANALYSIS_DIR
    / "conclusao_experimento1_2.json"
)

EXPECTED_WINDOWS = [25, 30, 50, 100]
EXPECTED_OVERLAPS = [25, 50, 75]
MIN_SESSION_FOLDS = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_manifest() -> dict:
    if not SOURCE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifesto da base de origem não encontrado:\n"
            f"{SOURCE_MANIFEST}"
        )

    with SOURCE_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def high_confidence_mask(
    series: pd.Series,
) -> pd.Series:

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    text = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return (
        numeric.eq(1)
        | text.isin(
            ["true", "1", "yes", "sim"]
        )
    )


def detect_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    metadata = {
        "_id",
        "user",
        "session_id",
        "window_id",
        "window_size",
        "requested_overlap_pct",
        "effective_overlap_pct",
        "step_events",
        "device_context",
        "device_high_confidence",
        "device_mouse_pct",
        "device_trackpad_pct",
        "window_start_ms",
        "window_end_ms",
        "start_timestamp",
        "end_timestamp",
        "target_user",
        "label",
        "class",
    }

    return [
        c
        for c in df.columns
        if c not in metadata
        and pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]


def main() -> None:
    TRACKPAD_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRACKPAD_ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "CYBELL-J — EXPERIMENTO 1.2 — "
        "ANÁLISE FINAL DO TRACKPAD"
    )
    print("=" * 100)

    print("\nRegras:")
    print("  Somente trackpad de alta confiança.")
    print("  Nenhum dado de mouse será incluído.")
    print("  Nenhum modelo será treinado nesta etapa.")

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Base de origem não encontrada:\n"
            f"{SOURCE_CSV}"
        )

    manifest = load_manifest()

    expected_hash = manifest.get(
        "dataset_device_sha256"
    )

    current_hash = sha256_file(
        SOURCE_CSV
    )

    print("\nVerificando integridade da fonte...")
    print(
        f"Hash esperado: {expected_hash}"
    )
    print(
        f"Hash atual:    {current_hash}"
    )

    if (
        expected_hash
        and current_hash != expected_hash
    ):
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
        c
        for c in required
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

    df[
        "requested_overlap_pct"
    ] = pd.to_numeric(
        df[
            "requested_overlap_pct"
        ],
        errors="coerce",
    )

    hc = high_confidence_mask(
        df["device_high_confidence"]
    )

    trackpad = df[
        df["device_context"].eq(
            "trackpad"
        )
        & hc
    ].copy()

    trackpad = trackpad.dropna(
        subset=[
            "user",
            "session_id",
            "window_id",
            "window_size",
            "requested_overlap_pct",
        ]
    ).reset_index(
        drop=True
    )

    trackpad["window_size"] = (
        trackpad[
            "window_size"
        ].astype(int)
    )

    trackpad[
        "requested_overlap_pct"
    ] = (
        trackpad[
            "requested_overlap_pct"
        ].astype(int)
    )

    if trackpad.empty:
        raise RuntimeError(
            "Nenhuma janela de trackpad "
            "de alta confiança encontrada."
        )

    # Garantia explícita: não existe mouse na base separada.
    if not (
        trackpad[
            "device_context"
        ].eq("trackpad")
    ).all():
        raise RuntimeError(
            "Foi detectado dado não-trackpad "
            "na base separada."
        )

    trackpad.to_csv(
        OUT_DATASET,
        index=False,
        encoding="utf-8",
    )

    print(
        f"\nJanelas separadas: "
        f"{len(trackpad):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Usuárias: "
        f"{trackpad['user'].nunique()}"
    )
    print(
        f"Sessões: "
        f"{trackpad['session_id'].nunique()}"
    )

    summary = (
        trackpad.groupby("user")
        .agg(
            janelas=(
                "window_id",
                "size",
            ),
            sessoes=(
                "session_id",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            "janelas",
            ascending=False,
        )
    )

    summary[
        "apto_3_folds"
    ] = (
        summary["sessoes"]
        >= MIN_SESSION_FOLDS
    )

    if "device_trackpad_pct" in trackpad.columns:
        trackpad[
            "device_trackpad_pct"
        ] = pd.to_numeric(
            trackpad[
                "device_trackpad_pct"
            ],
            errors="coerce",
        )

        quality = (
            trackpad.groupby("user")[
                "device_trackpad_pct"
            ]
            .agg(
                [
                    "mean",
                    "median",
                    "min",
                    "max",
                ]
            )
            .reset_index()
            .rename(
                columns={
                    "mean": (
                        "trackpad_pct_mean"
                    ),
                    "median": (
                        "trackpad_pct_median"
                    ),
                    "min": (
                        "trackpad_pct_min"
                    ),
                    "max": (
                        "trackpad_pct_max"
                    ),
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
        trackpad.groupby(
            [
                "user",
                "session_id",
            ]
        )
        .agg(
            janelas=(
                "window_id",
                "size",
            ),
            menor_janela=(
                "window_size",
                "min",
            ),
            maior_janela=(
                "window_size",
                "max",
            ),
            configuracoes=(
                "requested_overlap_pct",
                "count",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "user",
                "session_id",
            ]
        )
    )

    sessions.to_csv(
        OUT_SESSIONS,
        index=False,
        encoding="utf-8-sig",
    )

    feature_columns = (
        detect_feature_columns(
            trackpad
        )
    )

    feature_quality_rows = []

    for col in feature_columns:
        values = pd.to_numeric(
            trackpad[col],
            errors="coerce",
        ).replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        feature_quality_rows.append(
            {
                "feature": col,
                "missing_count": int(
                    values.isna().sum()
                ),
                "missing_pct": float(
                    values.isna().mean()
                    * 100.0
                ),
                "finite_count": int(
                    values.notna().sum()
                ),
                "mean": (
                    float(
                        values.mean()
                    )
                    if values.notna().any()
                    else np.nan
                ),
                "std": (
                    float(
                        values.std(
                            ddof=0
                        )
                    )
                    if values.notna().any()
                    else np.nan
                ),
                "median": (
                    float(
                        values.median()
                    )
                    if values.notna().any()
                    else np.nan
                ),
            }
        )

    feature_quality = (
        pd.DataFrame(
            feature_quality_rows
        )
        .sort_values(
            [
                "missing_pct",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    feature_quality.to_csv(
        OUT_FEATURE_QUALITY,
        index=False,
        encoding="utf-8-sig",
    )

    configs = (
        trackpad[
            [
                "window_size",
                "requested_overlap_pct",
            ]
        ]
        .drop_duplicates()
    )

    expected_configs = {
        (
            int(w),
            int(o),
        )
        for w in EXPECTED_WINDOWS
        for o in EXPECTED_OVERLAPS
    }

    observed_configs = {
        (
            int(row.window_size),
            int(
                row.requested_overlap_pct
            ),
        )
        for row in configs.itertuples()
    }

    users_3folds = (
        summary.loc[
            summary["sessoes"]
            >= MIN_SESSION_FOLDS,
            "user",
        ]
        .astype(str)
        .tolist()
    )

    training_feasible = (
        len(
            users_3folds
        )
        >= 2
    )

    conclusion = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "experiment": (
            "Experimento 1.2 - Trackpad"
        ),
        "source": str(
            SOURCE_CSV.relative_to(
                BASE_DIR
            )
        ),
        "source_sha256": current_hash,
        "selection": {
            "device_context": (
                "trackpad"
            ),
            "high_confidence_only": True,
            "mouse_included": False,
        },
        "input_analysis": {
            "windows": int(
                len(trackpad)
            ),
            "users": int(
                trackpad[
                    "user"
                ].nunique()
            ),
            "sessions": int(
                trackpad[
                    "session_id"
                ].nunique()
            ),
            "features_detected": int(
                len(
                    feature_columns
                )
            ),
            "observed_configurations": int(
                len(
                    observed_configs
                )
            ),
            "missing_configurations": [
                {
                    "window_size": int(
                        w
                    ),
                    "overlap_pct": int(
                        o
                    ),
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
        "recognition_model_training_feasible": (
            training_feasible
        ),
        "training_decision": (
            "not_train"
            if not training_feasible
            else "training_possible"
        ),
        "reason": (
            "Apenas uma participante possui pelo menos "
            "3 sessões independentes de trackpad. "
            "Não há base suficiente para treinamento e "
            "avaliação de reconhecimento entre pessoas "
            "com 3-fold por sessão sem introduzir mouse."
            if not training_feasible
            else
            "Há pelo menos duas participantes com "
            "3 sessões independentes."
        ),
        "normalization_if_future_training_becomes_possible": {
            "same_protocol_as_mouse": True,
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
        },
    }

    with OUT_DATASET_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "created_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "source": str(
                    SOURCE_CSV.relative_to(
                        BASE_DIR
                    )
                ),
                "source_sha256": (
                    current_hash
                ),
                "selection": (
                    "trackpad de alta confiança"
                ),
                "mouse_included": False,
                "rows": int(
                    len(trackpad)
                ),
                "users": sorted(
                    trackpad[
                        "user"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                ),
                "sessions": int(
                    trackpad[
                        "session_id"
                    ].nunique()
                ),
                "feature_columns": (
                    feature_columns
                ),
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

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
    print("RESUMO FINAL DO TRACKPAD")
    print("=" * 100)
    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nFeatures numéricas analisadas: "
        f"{len(feature_columns)}"
    )

    print(
        f"Configurações janela/overlap presentes: "
        f"{len(observed_configs)}/12"
    )

    print(
        "\nUsuárias com pelo menos "
        "3 sessões independentes:"
    )
    print(
        "  "
        + (
            ", ".join(
                users_3folds
            )
            if users_3folds
            else "nenhuma"
        )
    )

    print("\nDecisão do Experimento 1.2:")

    if training_feasible:
        print(
            "  Há base suficiente para "
            "avaliar treinamento específico."
        )
    else:
        print(
            "  NÃO treinar modelo de reconhecimento "
            "de trackpad nesta coleta."
        )
        print(
            "  Motivo: somente uma participante "
            "possui 3+ sessões independentes."
        )
        print(
            "  Não utilizar mouse para completar "
            "a base."
        )

    print("\nArquivos gerados:")
    print(f"  {OUT_DATASET}")
    print(f"  {OUT_DATASET_MANIFEST}")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_SESSIONS}")
    print(f"  {OUT_FEATURE_QUALITY}")
    print(f"  {OUT_CONCLUSION}")

    print(
        "\nEXPERIMENTO 1.2 DOCUMENTADO."
    )


if __name__ == "__main__":
    main()
