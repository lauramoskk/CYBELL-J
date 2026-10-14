"""
13_treinar_modelos_finais_cybellj.py

CYBELL-J — FASE 13 (FINAL)
TREINAMENTO, CALIBRAÇÃO E EMPACOTAMENTO DOS MODELOS DEFINITIVOS

Este é o último passo técnico de modelagem.

O que este script faz
---------------------
1. Reutiliza a base mouse de alta confiança já construída.
2. NÃO procura novas combinações nem novos algoritmos.
3. Usa a configuração congelada após os experimentos anteriores:
       janela = 50 eventos
       overlap solicitado = 75% (o dataset pode registrar 74%/76% como overlap efetivo)
       contexto = mouse
4. Usa um verificador 1:1 por identidade.
5. Mantém os vencedores clássicos definidos anteriormente:
       Julia104soup   -> Extra Trees + 20 s
       bianca         -> XGBoost    +  0 s
       lala           -> XGBoost    + 15 s
       laura_oliveira -> Extra Trees + 20 s
       lina           -> XGBoost    + 15 s
       yass           -> Random Forest + 5 s
6. Faz SOMENTE calibração final do limiar (threshold), com predições
   OOF e separação por session_id. Isso não é uma nova busca de modelo.
7. Reajusta cada verificador com todas as sessões disponíveis.
8. Salva os seis modelos finais em .pkl, junto com:
       - feature_names
       - threshold
       - horizonte temporal
       - configuração
       - métricas OOF de calibração
       - manifesto final
       - README de integração

Observação importante
---------------------
PyCaret foi usado na FASE EXPERIMENTAL para comparar/selecionar os
modelos clássicos. Nesta fase final usamos diretamente os estimadores
subjacentes (scikit-learn / XGBoost), o que deixa o artefato final mais
simples para integração e implantação.

Execução
--------
Ative o ambiente que contém PyCaret/XGBoost/scikit-learn:

    .venv_pycaret/Scripts/Activate.ps1
    python scripts_ml/13_treinar_modelos_finais_cybellj.py

Entrada esperada
----------------
data/mouse_dataset/dataset_device_high_confidence.csv

Saída
-----
data/modelos_finais_cybellj/
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from xgboost import XGBClassifier
except Exception as exc:
    raise SystemExit(
        "\nERRO: xgboost não está disponível no ambiente.\n"
        "Ative .venv_pycaret e execute o script novamente.\n"
    ) from exc


# ============================================================
# RECURSOS
# ============================================================

CPU_THREADS = 2

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "modelos_finais_cybellj"
)

MODELS_DIR = OUTPUT_DIR / "modelos"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_PRED_PATH = (
    OUTPUT_DIR
    / "predicoes_oof_calibracao.csv"
)

CALIBRATION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "resumo_calibracao_final.csv"
)

FINAL_CONFIG_PATH = (
    OUTPUT_DIR
    / "configuracao_final.csv"
)

FEATURES_PATH = (
    OUTPUT_DIR
    / "feature_names.json"
)

THRESHOLDS_PATH = (
    OUTPUT_DIR
    / "thresholds_finais.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "manifesto_final_cybellj.json"
)

README_PATH = (
    OUTPUT_DIR
    / "README_MODELOS_FINAIS.txt"
)


# ============================================================
# CONFIGURAÇÃO CONGELADA — NÃO É MAIS UMA BUSCA DE MODELO
# ============================================================

RANDOM_STATE = 42
N_FOLDS = 3

WINDOW_SIZE = 50
OVERLAP_PERCENT = 75

IMPOSTOR_TO_GENUINE_TRAIN_RATIO = 2.0

TEMPORAL_REQUIRE_FULL_HORIZON = True
TEMPORAL_MIN_WINDOWS = 2

FINAL_PROFILES = {
    "Julia104soup": {
        "candidate": "ExtraTrees",
        "horizon_s": 20,
        "reference_eer": 0.118991,
    },
    "bianca": {
        "candidate": "XGBoost",
        "horizon_s": 0,
        "reference_eer": 0.277346,
    },
    "lala": {
        "candidate": "XGBoost",
        "horizon_s": 15,
        "reference_eer": 0.253519,
    },
    "laura_oliveira": {
        "candidate": "ExtraTrees",
        "horizon_s": 20,
        "reference_eer": 0.150705,
    },
    "lina": {
        "candidate": "XGBoost",
        "horizon_s": 15,
        "reference_eer": 0.051685,
    },
    "yass": {
        "candidate": "RandomForest",
        "horizon_s": 5,
        "reference_eer": 0.000000,
    },
}


# ============================================================
# COLUNAS / FEATURES
# ============================================================

USER_CANDIDATES = [
    "user",
    "username",
    "target_user",
    "protected_user",
]

SESSION_CANDIDATES = [
    "session_id",
    "session",
    "sessionid",
]

WINDOW_SIZE_CANDIDATES = [
    "window_size",
    "window_events",
    "n_events",
    "window_n_events",
]

OVERLAP_CANDIDATES = [
    "overlap_percent",
    "overlap_pct",
    "overlap",
    "effective_overlap_percent",
    "effective_overlap_pct",
]

DEVICE_CONTEXT_CANDIDATES = [
    "device_context",
    "window_device",
    "context_device",
]

END_TIME_CANDIDATES = [
    "window_end_ms",
    "window_end_timestamp",
    "end_timestamp",
    "window_end_time",
    "end_time",
    "timestamp_end",
    "ts_end",
    "end_ts",
]

START_TIME_CANDIDATES = [
    "window_start_ms",
    "window_start_timestamp",
    "start_timestamp",
    "window_start_time",
    "start_time",
    "timestamp_start",
    "ts_start",
    "start_ts",
]

EXACT_METADATA_EXCLUSIONS = {
    "user",
    "username",
    "target_user",
    "protected_user",
    "actual_user",
    "session_id",
    "session",
    "sessionid",
    "window_size",
    "window_events",
    "n_events",
    "window_n_events",
    "overlap_percent",
    "overlap_pct",
    "overlap",
    "effective_overlap_percent",
    "effective_overlap_pct",
    "device_context",
    "window_device",
    "context_device",
    "device_type",
    "target_user_device",
    "device_confidence",
    "high_confidence",
    "mouse_pct",
    "trackpad_pct",
    "device_mouse_pct",
    "device_trackpad_pct",
    "mouse_percentage",
    "trackpad_percentage",
    "window_start_ms",
    "window_end_ms",
    "window_start_timestamp",
    "window_end_timestamp",
    "start_timestamp",
    "end_timestamp",
    "window_start_time",
    "window_end_time",
    "start_time",
    "end_time",
    "timestamp_start",
    "timestamp_end",
    "ts_start",
    "ts_end",
    "start_ts",
    "end_ts",
    "fold",
    "split",
    "target",
    "label",
    "class",
}


# ============================================================
# UTILITÁRIOS
# ============================================================

def stable_seed(text: str) -> int:
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        RANDOM_STATE
        + int(
            digest[:8],
            16,
        )
    ) % (
        2**32
        - 1
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def first_existing(
    columns,
    candidates,
):
    lookup = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[
                candidate.lower()
            ]

    return None


def clean_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def normalize_overlap_value(series: pd.Series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.empty:
        return values

    # Alguns pipelines salvam 75; outros 0.75.
    median_abs = float(
        np.nanmedian(
            np.abs(
                finite.to_numpy()
            )
        )
    )

    if median_abs <= 1.5:
        return values * 100.0

    return values


# ============================================================
# TEMPO
# ============================================================

def detect_time_column(df: pd.DataFrame):
    col = first_existing(
        df.columns,
        END_TIME_CANDIDATES,
    )

    if col is not None:
        return col

    # Busca conservadora por nomes de coluna.
    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for lname, original in lower_map.items():
        if (
            "end" in lname
            and (
                "time" in lname
                or "timestamp" in lname
                or lname.endswith("_ts")
            )
        ):
            return original

    # Como último recurso, usar início da janela.
    col = first_existing(
        df.columns,
        START_TIME_CANDIDATES,
    )

    if col is not None:
        return col

    for lname, original in lower_map.items():
        if (
            "start" in lname
            and (
                "time" in lname
                or "timestamp" in lname
                or lname.endswith("_ts")
            )
        ):
            return original

    return None


def convert_time_to_ms(series: pd.Series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        raise RuntimeError(
            "A coluna temporal existe, mas não contém valores numéricos válidos."
        )

    median_abs = float(
        np.nanmedian(
            np.abs(
                finite
            )
        )
    )

    # epoch em ns
    if median_abs > 1e17:
        factor = 1e-6

    # epoch em microssegundos
    elif median_abs > 1e14:
        factor = 1e-3

    # epoch em milissegundos
    elif median_abs > 1e11:
        factor = 1.0

    # epoch em segundos
    elif median_abs > 1e8:
        factor = 1000.0

    # relativo em segundos costuma ficar pequeno.
    else:
        # Se os passos típicos são muito pequenos, assumir segundos.
        diffs = np.diff(
            np.sort(
                np.unique(
                    finite[: min(
                        len(finite),
                        10000,
                    )]
                )
            )
        )

        diffs = diffs[
            diffs > 0
        ]

        if (
            diffs.size > 0
            and float(
                np.nanmedian(
                    diffs
                )
            ) < 10
        ):
            factor = 1000.0
        else:
            factor = 1.0

    out = np.rint(
        values
        * factor
    )

    return out.astype(
        "float64"
    )


# ============================================================
# CARREGAMENTO E FILTRO
# ============================================================

def load_final_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset não encontrado:\n{DATASET_PATH}\n"
        )

    print(
        f"\nCarregando:\n{DATASET_PATH}"
    )

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False,
    )

    print(
        f"Linhas originais: "
        f"{len(df):,}"
        .replace(
            ",",
            ".",
        )
    )

    user_col = first_existing(
        df.columns,
        USER_CANDIDATES,
    )

    session_col = first_existing(
        df.columns,
        SESSION_CANDIDATES,
    )

    if user_col is None:
        raise RuntimeError(
            "Não consegui localizar a coluna de usuário.\n"
            f"Colunas: {df.columns.tolist()}"
        )

    if session_col is None:
        raise RuntimeError(
            "Não consegui localizar session_id.\n"
            f"Colunas: {df.columns.tolist()}"
        )

    window_col = first_existing(
        df.columns,
        WINDOW_SIZE_CANDIDATES,
    )

    overlap_col = first_existing(
        df.columns,
        OVERLAP_CANDIDATES,
    )

    device_col = first_existing(
        df.columns,
        DEVICE_CONTEXT_CANDIDATES,
    )

    if window_col is not None:
        window_values = pd.to_numeric(
            df[
                window_col
            ],
            errors="coerce",
        )

        df = df.loc[
            window_values.eq(
                WINDOW_SIZE
            )
        ].copy()

        print(
            f"Filtro janela={WINDOW_SIZE}: "
            f"{len(df):,}"
            .replace(
                ",",
                ".",
            )
        )
    else:
        print(
            "AVISO: coluna de window_size não encontrada; "
            "o arquivo será assumido como já filtrado."
        )

    if overlap_col is not None:
        overlap_values = normalize_overlap_value(
            df[
                overlap_col
            ]
        )

        available_overlap = np.sort(
            pd.Series(
                overlap_values
            )
            .dropna()
            .unique()
            .astype(float)
        )

        if len(available_overlap) == 0:
            raise RuntimeError(
                "A coluna de overlap existe, mas não possui valores válidos."
            )

        # O gerador do dataset registra o overlap EFETIVO.
        # Para janela de 50 eventos, overlap solicitado de 75% exige
        # stride inteiro. Ex.: stride=12 -> overlap efetivo=76%.
        # Por isso não devemos exigir exatamente 75.000%.
        nearest_overlap = float(
            available_overlap[
                np.argmin(
                    np.abs(
                        available_overlap
                        - OVERLAP_PERCENT
                    )
                )
            ]
        )

        distance_to_target = abs(
            nearest_overlap
            - OVERLAP_PERCENT
        )

        print(
            "Overlaps efetivos disponíveis para "
            f"janela={WINDOW_SIZE}: "
            + ", ".join(
                f"{v:.4g}%"
                for v in available_overlap
            )
        )

        if distance_to_target > 5.0:
            raise RuntimeError(
                f"Nenhum overlap efetivo compatível com o alvo "
                f"{OVERLAP_PERCENT}% foi encontrado. "
                f"O mais próximo foi {nearest_overlap:.4g}%."
            )

        mask_overlap = np.isclose(
            overlap_values,
            nearest_overlap,
            atol=1e-9,
        )

        df = df.loc[
            mask_overlap
        ].copy()

        print(
            f"Filtro overlap solicitado={OVERLAP_PERCENT}% "
            f"(efetivo no dataset={nearest_overlap:.4g}%): "
            f"{len(df):,}"
            .replace(
                ",",
                ".",
            )
        )
    else:
        print(
            "AVISO: coluna de overlap não encontrada; "
            "o arquivo será assumido como já filtrado."
        )

    if device_col is not None:
        dev = (
            df[
                device_col
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if (
            dev.eq(
                "mouse"
            )
        ).any():
            df = df.loc[
                dev.eq(
                    "mouse"
                )
            ].copy()

            print(
                f"Filtro device_context=mouse: "
                f"{len(df):,}"
                .replace(
                    ",",
                    ".",
                )
            )

    df[
        user_col
    ] = (
        df[
            user_col
        ]
        .astype(str)
        .str.strip()
    )

    df[
        session_col
    ] = (
        df[
            session_col
        ]
        .astype(str)
        .str.strip()
    )

    present_users = set(
        df[
            user_col
        ].unique()
    )

    missing_users = [
        user
        for user in FINAL_PROFILES
        if user not in present_users
    ]

    if missing_users:
        raise RuntimeError(
            "Perfis finais ausentes após os filtros: "
            + ", ".join(
                missing_users
            )
        )

    df = df.loc[
        df[
            user_col
        ].isin(
            FINAL_PROFILES.keys()
        )
    ].copy()

    time_col = detect_time_column(
        df
    )

    if time_col is None:
        raise RuntimeError(
            "\nNão consegui localizar a coluna de tempo da janela.\n"
            "Ela é necessária para calibrar 5/15/20 segundos.\n"
            "Colunas encontradas:\n"
            + "\n".join(
                map(
                    str,
                    df.columns,
                )
            )
        )

    df[
        "_time_ms"
    ] = convert_time_to_ms(
        df[
            time_col
        ]
    )

    df = df.dropna(
        subset=[
            user_col,
            session_col,
            "_time_ms",
        ]
    ).copy()

    df[
        "_time_ms"
    ] = np.rint(
        df[
            "_time_ms"
        ]
    ).astype(
        "int64"
    )

    df = df.sort_values(
        [
            user_col,
            session_col,
            "_time_ms",
        ]
    ).reset_index(
        drop=True
    )

    return (
        df,
        user_col,
        session_col,
        time_col,
    )


# ============================================================
# FEATURES — MESMA FILOSOFIA DO PIPELINE CLÁSSICO
# ============================================================

def choose_features(
    df: pd.DataFrame,
    user_col: str,
    session_col: str,
    time_col: str,
):
    excluded = {
        str(x).lower()
        for x in EXACT_METADATA_EXCLUSIONS
    }

    excluded.add(
        str(
            user_col
        ).lower()
    )

    excluded.add(
        str(
            session_col
        ).lower()
    )

    excluded.add(
        str(
            time_col
        ).lower()
    )

    excluded.add(
        "_time_ms"
    )

    features = []

    for col in df.columns:
        lname = str(
            col
        ).lower()

        if lname in excluded:
            continue

        # Bloqueia metadata derivada do rótulo de dispositivo.
        if (
            "device_confidence" in lname
            or "high_confidence" in lname
            or "trackpad_pct" in lname
            or "mouse_pct" in lname
            or "trackpad_percentage" in lname
            or "mouse_percentage" in lname
            or lname.startswith(
                "target_"
            )
            or lname.endswith(
                "_target"
            )
        ):
            continue

        if pd.api.types.is_numeric_dtype(
            df[
                col
            ]
        ):
            features.append(
                col
            )

    if len(
        features
    ) < 20:
        raise RuntimeError(
            f"Apenas {len(features)} features numéricas foram encontradas. "
            "Isso indica que o layout do dataset mudou."
        )

    print(
        f"\nFeatures comportamentais selecionadas: "
        f"{len(features)}"
    )

    if len(
        features
    ) != 89:
        print(
            "AVISO: o pipeline experimental tinha 89 features. "
            f"Este arquivo produziu {len(features)}. "
            "O script continuará, mas salvará a lista exata para auditoria."
        )

    with FEATURES_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "n_features": len(
                    features
                ),
                "features": features,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    return features


def numeric_matrix(
    df: pd.DataFrame,
    features,
):
    X = (
        df[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    # Preenchimento robusto sem usar o rótulo.
    medians = X.median(
        axis=0,
        numeric_only=True,
    )

    X = X.fillna(
        medians
    ).fillna(
        0.0
    )

    return (
        X.astype(
            np.float32
        ),
        medians.astype(
            np.float32
        ),
    )


# ============================================================
# FOLDS POR SESSION_id
# ============================================================

def build_session_folds(
    df: pd.DataFrame,
    user_col: str,
    session_col: str,
):
    sessions_by_fold = [
        set()
        for _ in range(
            N_FOLDS
        )
    ]

    for user in sorted(
        FINAL_PROFILES.keys()
    ):
        sessions = sorted(
            df.loc[
                df[
                    user_col
                ].eq(
                    user
                ),
                session_col,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(
            sessions
        ) < N_FOLDS:
            raise RuntimeError(
                f"{user}: somente {len(sessions)} sessões. "
                f"São necessárias pelo menos {N_FOLDS}."
            )

        rng = np.random.default_rng(
            stable_seed(
                f"folds|{user}"
            )
        )

        rng.shuffle(
            sessions
        )

        for i, session_id in enumerate(
            sessions
        ):
            sessions_by_fold[
                i
                % N_FOLDS
            ].add(
                session_id
            )

    folds = []

    all_sessions = set(
        df[
            session_col
        ].astype(str).unique()
    )

    for fold_idx in range(
        N_FOLDS
    ):
        test_sessions = sessions_by_fold[
            fold_idx
        ]

        folds.append(
            {
                "fold": fold_idx + 1,
                "test_sessions": test_sessions,
                "train_sessions": all_sessions - test_sessions,
            }
        )

    return folds


# ============================================================
# AMOSTRAGEM DE IMPOSTORES — MESMA RAZÃO 2:1
# ============================================================

def sample_impostors(
    df: pd.DataFrame,
    candidate_indices: np.ndarray,
    n_genuine: int,
    protected_user: str,
    fold_tag: str,
    user_col: str,
):
    target = int(
        round(
            n_genuine
            * IMPOSTOR_TO_GENUINE_TRAIN_RATIO
        )
    )

    if (
        len(
            candidate_indices
        )
        <= target
    ):
        return candidate_indices

    pool = df.loc[
        candidate_indices,
        [
            user_col
        ],
    ].copy()

    pool[
        "_idx"
    ] = candidate_indices

    impostor_users = sorted(
        pool[
            user_col
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    chosen = []
    remaining = target

    for pos, imp_user in enumerate(
        impostor_users
    ):
        idx = pool.loc[
            pool[
                user_col
            ]
            .astype(str)
            .eq(
                imp_user
            ),
            "_idx",
        ].to_numpy(
            dtype=np.int64
        )

        groups_left = (
            len(
                impostor_users
            )
            - pos
        )

        take = min(
            len(
                idx
            ),
            max(
                1,
                remaining
                // groups_left,
            ),
        )

        rng = np.random.default_rng(
            stable_seed(
                f"imp|{protected_user}|{fold_tag}|{imp_user}"
            )
        )

        chosen.append(
            rng.choice(
                idx,
                size=take,
                replace=False,
            )
        )

        remaining -= take

    sampled = np.concatenate(
        chosen
    ).astype(
        np.int64
    )

    if len(
        sampled
    ) > target:
        rng = np.random.default_rng(
            stable_seed(
                f"imp-final-trim|{protected_user}|{fold_tag}"
            )
        )

        sampled = rng.choice(
            sampled,
            size=target,
            replace=False,
        )

    return np.asarray(
        sampled,
        dtype=np.int64,
    )


# ============================================================
# MODELOS CONGELADOS
# ============================================================

def build_estimator(
    candidate: str,
    seed: int,
):
    if candidate == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=CPU_THREADS,
            max_features="sqrt",
            class_weight=None,
        )

    if candidate == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=CPU_THREADS,
            max_features="sqrt",
            class_weight=None,
        )

    if candidate == "XGBoost":
        return XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.90,
            colsample_bytree=0.90,
            max_bin=128,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=CPU_THREADS,
            tree_method="hist",
            device="cpu",
            verbosity=0,
        )

    raise ValueError(
        f"Candidato não reconhecido: {candidate}"
    )


# ============================================================
# MÉTRICAS
# ============================================================

def compute_eer(
    y_true,
    scores,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if len(
        np.unique(
            y_true
        )
    ) < 2:
        raise RuntimeError(
            "EER exige genuínos e impostores."
        )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    idx = int(
        np.nanargmin(
            np.abs(
                fpr
                - fnr
            )
        )
    )

    far = float(
        fpr[
            idx
        ]
    )

    frr = float(
        fnr[
            idx
        ]
    )

    return {
        "eer": float(
            (
                far
                + frr
            )
            / 2.0
        ),
        "threshold": float(
            thresholds[
                idx
            ]
        ),
        "far": far,
        "frr": frr,
    }


def compute_metrics(
    y_true,
    scores,
    threshold,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    pred = (
        scores
        >= threshold
    ).astype(
        np.int8
    )

    return {
        "auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
    }


# ============================================================
# AGREGAÇÃO TEMPORAL
# ============================================================

def aggregate_session(
    session: pd.DataFrame,
    horizon_s: int,
):
    g = session.sort_values(
        [
            "_time_ms",
            "_row_index",
        ]
    ).reset_index(
        drop=True
    )

    times = g[
        "_time_ms"
    ].to_numpy(
        dtype=np.int64
    )

    scores = g[
        "score_genuine"
    ].to_numpy(
        dtype=np.float64
    )

    labels = g[
        "y_true_genuine"
    ].to_numpy(
        dtype=np.int8
    )

    if horizon_s == 0:
        return pd.DataFrame(
            {
                "y_true_genuine": labels,
                "score_eval": scores,
                "time_ms": times,
                "n_windows": np.ones(
                    len(
                        g
                    ),
                    dtype=np.int32,
                ),
            }
        )

    horizon_ms = int(
        horizon_s
        * 1000
    )

    cumulative = np.concatenate(
        [
            np.array(
                [
                    0.0
                ],
                dtype=np.float64,
            ),
            np.cumsum(
                scores,
                dtype=np.float64,
            ),
        ]
    )

    first_time = int(
        times[
            0
        ]
    )

    left = 0

    out_y = []
    out_score = []
    out_time = []
    out_n = []

    for i in range(
        len(
            g
        )
    ):
        now = int(
            times[
                i
            ]
        )

        while (
            left
            < i
            and (
                now
                - int(
                    times[
                        left
                    ]
                )
                > horizon_ms
            )
        ):
            left += 1

        n = (
            i
            - left
            + 1
        )

        if n < TEMPORAL_MIN_WINDOWS:
            continue

        if (
            TEMPORAL_REQUIRE_FULL_HORIZON
            and (
                now
                - first_time
                < horizon_ms
            )
        ):
            continue

        score = (
            cumulative[
                i + 1
            ]
            - cumulative[
                left
            ]
        ) / n

        out_y.append(
            int(
                labels[
                    i
                ]
            )
        )

        out_score.append(
            float(
                score
            )
        )

        out_time.append(
            now
        )

        out_n.append(
            int(
                n
            )
        )

    return pd.DataFrame(
        {
            "y_true_genuine": out_y,
            "score_eval": out_score,
            "time_ms": out_time,
            "n_windows": out_n,
        }
    )


def aggregate_oof(
    pred_df: pd.DataFrame,
    horizon_s: int,
):
    parts = []

    for (
        fold,
        session_id,
    ), session in pred_df.groupby(
        [
            "fold",
            "session_id",
        ],
        sort=False,
    ):
        part = aggregate_session(
            session,
            horizon_s=horizon_s,
        )

        if part.empty:
            continue

        part[
            "fold"
        ] = int(
            fold
        )

        part[
            "session_id"
        ] = str(
            session_id
        )

        parts.append(
            part
        )

    if not parts:
        raise RuntimeError(
            f"Nenhuma decisão restou após horizonte {horizon_s}s."
        )

    aggregated = pd.concat(
        parts,
        ignore_index=True,
    )

    if (
        aggregated[
            "y_true_genuine"
        ].nunique()
        < 2
    ):
        raise RuntimeError(
            f"Horizonte {horizon_s}s ficou com apenas uma classe."
        )

    return aggregated


# ============================================================
# CALIBRAÇÃO OOF FIXA
# ============================================================

def calibrate_profile(
    df: pd.DataFrame,
    X: pd.DataFrame,
    protected_user: str,
    config: dict,
    folds,
    user_col: str,
    session_col: str,
):
    candidate = config[
        "candidate"
    ]

    horizon_s = int(
        config[
            "horizon_s"
        ]
    )

    prediction_parts = []

    for fold_info in folds:
        fold = int(
            fold_info[
                "fold"
            ]
        )

        train_session_set = fold_info[
            "train_sessions"
        ]

        test_session_set = fold_info[
            "test_sessions"
        ]

        train_mask = (
            df[
                session_col
            ]
            .astype(str)
            .isin(
                train_session_set
            )
        )

        test_mask = (
            df[
                session_col
            ]
            .astype(str)
            .isin(
                test_session_set
            )
        )

        genuine_train_idx = np.flatnonzero(
            (
                train_mask
                & df[
                    user_col
                ].eq(
                    protected_user
                )
            ).to_numpy()
        ).astype(
            np.int64
        )

        impostor_pool_idx = np.flatnonzero(
            (
                train_mask
                & ~df[
                    user_col
                ].eq(
                    protected_user
                )
            ).to_numpy()
        ).astype(
            np.int64
        )

        test_idx = np.flatnonzero(
            test_mask.to_numpy()
        ).astype(
            np.int64
        )

        sampled_imp_idx = sample_impostors(
            df=df,
            candidate_indices=impostor_pool_idx,
            n_genuine=len(
                genuine_train_idx
            ),
            protected_user=protected_user,
            fold_tag=f"fold{fold}",
            user_col=user_col,
        )

        train_idx = np.concatenate(
            [
                genuine_train_idx,
                sampled_imp_idx,
            ]
        )

        rng = np.random.default_rng(
            stable_seed(
                f"train-shuffle|{protected_user}|fold{fold}"
            )
        )

        rng.shuffle(
            train_idx
        )

        y_train = (
            df.loc[
                train_idx,
                user_col,
            ]
            .eq(
                protected_user
            )
            .astype(
                np.int8
            )
            .to_numpy()
        )

        y_test = (
            df.loc[
                test_idx,
                user_col,
            ]
            .eq(
                protected_user
            )
            .astype(
                np.int8
            )
            .to_numpy()
        )

        model = build_estimator(
            candidate=candidate,
            seed=stable_seed(
                f"model|{protected_user}|fold{fold}"
            ),
        )

        model.fit(
            X.iloc[
                train_idx
            ],
            y_train,
        )

        scores = model.predict_proba(
            X.iloc[
                test_idx
            ]
        )[
            :,
            1,
        ].astype(
            np.float64
        )

        fold_pred = pd.DataFrame(
            {
                "protected_user": protected_user,
                "candidate": candidate,
                "horizon_s": horizon_s,
                "fold": fold,
                "session_id": df.loc[
                    test_idx,
                    session_col,
                ].astype(
                    str
                ).to_numpy(),
                "actual_user": df.loc[
                    test_idx,
                    user_col,
                ].astype(
                    str
                ).to_numpy(),
                "_time_ms": df.loc[
                    test_idx,
                    "_time_ms",
                ].to_numpy(
                    dtype=np.int64
                ),
                "_row_index": test_idx,
                "y_true_genuine": y_test,
                "score_genuine": scores,
            }
        )

        prediction_parts.append(
            fold_pred
        )

        print(
            f"      fold {fold}: "
            f"treino={len(train_idx):,} "
            f"teste={len(test_idx):,}"
            .replace(
                ",",
                ".",
            )
        )

        del (
            model,
            scores,
            y_train,
            y_test,
            train_idx,
            test_idx,
            genuine_train_idx,
            sampled_imp_idx,
            impostor_pool_idx,
            fold_pred,
        )

        gc.collect()

    pred_df = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    temporal = aggregate_oof(
        pred_df=pred_df,
        horizon_s=horizon_s,
    )

    eer_info = compute_eer(
        temporal[
            "y_true_genuine"
        ].to_numpy(),
        temporal[
            "score_eval"
        ].to_numpy(),
    )

    metrics = compute_metrics(
        temporal[
            "y_true_genuine"
        ].to_numpy(),
        temporal[
            "score_eval"
        ].to_numpy(),
        threshold=eer_info[
            "threshold"
        ],
    )

    coverage = float(
        len(
            temporal
        )
        / len(
            pred_df
        )
    )

    summary = {
        "protected_user": protected_user,
        "candidate": candidate,
        "window_size": WINDOW_SIZE,
        "overlap_percent": OVERLAP_PERCENT,
        "horizon_s": horizon_s,
        "threshold_final": float(
            eer_info[
                "threshold"
            ]
        ),
        "eer_oof_calibration": float(
            eer_info[
                "eer"
            ]
        ),
        "far_oof_calibration": float(
            eer_info[
                "far"
            ]
        ),
        "frr_oof_calibration": float(
            eer_info[
                "frr"
            ]
        ),
        "auc_oof_calibration": float(
            metrics[
                "auc"
            ]
        ),
        "f1_oof_calibration": float(
            metrics[
                "f1"
            ]
        ),
        "balanced_accuracy_oof_calibration": float(
            metrics[
                "balanced_accuracy"
            ]
        ),
        "precision_oof_calibration": float(
            metrics[
                "precision"
            ]
        ),
        "recall_oof_calibration": float(
            metrics[
                "recall"
            ]
        ),
        "n_oof_windows": int(
            len(
                pred_df
            )
        ),
        "n_temporal_decisions": int(
            len(
                temporal
            )
        ),
        "coverage_temporal": coverage,
        "reference_eer_previous_experiment": float(
            config[
                "reference_eer"
            ]
        ),
    }

    return (
        pred_df,
        temporal,
        summary,
    )


# ============================================================
# TREINO DEFINITIVO EM TODAS AS SESSÕES
# ============================================================

def train_final_model(
    df: pd.DataFrame,
    X: pd.DataFrame,
    protected_user: str,
    config: dict,
    user_col: str,
):
    genuine_idx = np.flatnonzero(
        df[
            user_col
        ]
        .eq(
            protected_user
        )
        .to_numpy()
    ).astype(
        np.int64
    )

    impostor_pool_idx = np.flatnonzero(
        ~df[
            user_col
        ]
        .eq(
            protected_user
        )
        .to_numpy()
    ).astype(
        np.int64
    )

    sampled_imp_idx = sample_impostors(
        df=df,
        candidate_indices=impostor_pool_idx,
        n_genuine=len(
            genuine_idx
        ),
        protected_user=protected_user,
        fold_tag="FINAL_ALL_SESSIONS",
        user_col=user_col,
    )

    train_idx = np.concatenate(
        [
            genuine_idx,
            sampled_imp_idx,
        ]
    )

    rng = np.random.default_rng(
        stable_seed(
            f"final-shuffle|{protected_user}"
        )
    )

    rng.shuffle(
        train_idx
    )

    y_train = (
        df.loc[
            train_idx,
            user_col,
        ]
        .eq(
            protected_user
        )
        .astype(
            np.int8
        )
        .to_numpy()
    )

    model = build_estimator(
        candidate=config[
            "candidate"
        ],
        seed=stable_seed(
            f"final-model|{protected_user}"
        ),
    )

    model.fit(
        X.iloc[
            train_idx
        ],
        y_train,
    )

    return (
        model,
        {
            "n_train_total": int(
                len(
                    train_idx
                )
            ),
            "n_train_genuine": int(
                len(
                    genuine_idx
                )
            ),
            "n_train_impostor": int(
                len(
                    sampled_imp_idx
                )
            ),
        },
    )


# ============================================================
# README
# ============================================================

def write_readme(
    summaries: pd.DataFrame,
    n_features: int,
):
    lines = [
        "CYBELL-J — MODELOS FINAIS",
        "=" * 72,
        "",
        "Uso conceitual de cada verificador:",
        "",
        "1. Receber os eventos de mouse.",
        f"2. Reproduzir EXATAMENTE o pipeline que gera as {n_features} features.",
        f"3. Formar janelas de {WINDOW_SIZE} eventos com overlap de {OVERLAP_PERCENT}%.",
        "4. Carregar o .pkl da identidade logada.",
        "5. Executar model.predict_proba(X)[:, 1].",
        "6. Se horizon_s > 0, calcular média móvel retrospectiva dos scores",
        "   durante o horizonte configurado.",
        "7. Comparar score agregado com threshold_final.",
        "8. score >= threshold_final -> comportamento compatível.",
        "9. score < threshold_final  -> sinalizar comportamento suspeito.",
        "",
        "IMPORTANTE:",
        "- Os thresholds foram calibrados com predições OOF separadas por session_id.",
        "- O threshold de pesquisa não deve ser alterado silenciosamente em produção.",
        "- O pipeline de features deve ser idêntico ao usado neste treinamento.",
        "- PyCaret foi usado na seleção experimental; os artefatos finais usam",
        "  diretamente scikit-learn/XGBoost para facilitar implantação.",
        "",
        "CONFIGURAÇÃO FINAL:",
        "",
    ]

    for _, row in summaries.iterrows():
        lines.append(
            (
                f"{row['protected_user']}: "
                f"{row['candidate']} | "
                f"horizonte={int(row['horizon_s'])}s | "
                f"threshold={row['threshold_final']:.8f} | "
                f"EER OOF calibração={row['eer_oof_calibration']:.6f}"
            )
        )

    README_PATH.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "="
        * 112
    )

    print(
        "CYBELL-J — FASE 13 FINAL — "
        "TREINAMENTO DOS MODELOS DEFINITIVOS"
    )

    print(
        "="
        * 112
    )

    print(
        "\nSem busca de hiperparâmetros."
        "\nSem novos algoritmos."
        "\nSem LSTM."
        "\nConfiguração clássica já congelada."
    )

    seed_everything(
        RANDOM_STATE
    )

    (
        df,
        user_col,
        session_col,
        time_col,
    ) = load_final_dataset()

    features = choose_features(
        df=df,
        user_col=user_col,
        session_col=session_col,
        time_col=time_col,
    )

    X, medians = numeric_matrix(
        df=df,
        features=features,
    )

    folds = build_session_folds(
        df=df,
        user_col=user_col,
        session_col=session_col,
    )

    print(
        "\nSessões / janelas da configuração final:"
    )

    table = (
        df.groupby(
            user_col
        )
        .agg(
            windows=(
                session_col,
                "size",
            ),
            sessions=(
                session_col,
                "nunique",
            ),
        )
        .reset_index()
    )

    print(
        table.to_string(
            index=False
        )
    )

    if CALIBRATION_PRED_PATH.exists():
        CALIBRATION_PRED_PATH.unlink()

    calibration_summaries = []
    thresholds = {}

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "1) CALIBRAÇÃO FINAL OOF — "
        "MODELOS JÁ ESCOLHIDOS"
    )

    print(
        "="
        * 112
    )

    for pos, (
        protected_user,
        config,
    ) in enumerate(
        FINAL_PROFILES.items(),
        start=1,
    ):
        print(
            f"\n[{pos}/{len(FINAL_PROFILES)}] "
            f"{protected_user} | "
            f"{config['candidate']} | "
            f"horizonte={config['horizon_s']}s"
        )

        (
            pred_df,
            temporal_df,
            summary,
        ) = calibrate_profile(
            df=df,
            X=X,
            protected_user=protected_user,
            config=config,
            folds=folds,
            user_col=user_col,
            session_col=session_col,
        )

        pred_df.to_csv(
            CALIBRATION_PRED_PATH,
            mode="a",
            header=not CALIBRATION_PRED_PATH.exists(),
            index=False,
            encoding="utf-8-sig",
        )

        calibration_summaries.append(
            summary
        )

        thresholds[
            protected_user
        ] = {
            "threshold_final": float(
                summary[
                    "threshold_final"
                ]
            ),
            "horizon_s": int(
                config[
                    "horizon_s"
                ]
            ),
            "candidate": config[
                "candidate"
            ],
            "window_size": WINDOW_SIZE,
            "overlap_percent": OVERLAP_PERCENT,
        }

        print(
            f"   CALIBRAÇÃO | "
            f"EER={summary['eer_oof_calibration']:.4f} | "
            f"AUC={summary['auc_oof_calibration']:.4f} | "
            f"F1={summary['f1_oof_calibration']:.4f} | "
            f"threshold={summary['threshold_final']:.6f}"
        )

        del (
            pred_df,
            temporal_df,
        )

        gc.collect()

    summary_df = pd.DataFrame(
        calibration_summaries
    )

    summary_df.to_csv(
        CALIBRATION_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with THRESHOLDS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            thresholds,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "2) TREINANDO OS 6 MODELOS DEFINITIVOS "
        "COM TODAS AS SESSÕES"
    )

    print(
        "="
        * 112
    )

    final_rows = []

    for pos, (
        protected_user,
        config,
    ) in enumerate(
        FINAL_PROFILES.items(),
        start=1,
    ):
        print(
            f"\n[{pos}/{len(FINAL_PROFILES)}] "
            f"Treino final: {protected_user}"
        )

        model, train_info = train_final_model(
            df=df,
            X=X,
            protected_user=protected_user,
            config=config,
            user_col=user_col,
        )

        calibration_row = summary_df.loc[
            summary_df[
                "protected_user"
            ].eq(
                protected_user
            )
        ].iloc[
            0
        ]

        package = {
            "project": "CYBELL-J",
            "task": (
                "continuous_authentication_1_to_1"
            ),
            "protected_user": protected_user,
            "candidate": config[
                "candidate"
            ],
            "model": model,
            "feature_names": features,
            "feature_medians": medians.to_dict(),
            "window_size": WINDOW_SIZE,
            "overlap_percent": OVERLAP_PERCENT,
            "device_context": "mouse",
            "horizon_s": int(
                config[
                    "horizon_s"
                ]
            ),
            "threshold_final": float(
                calibration_row[
                    "threshold_final"
                ]
            ),
            "threshold_source": (
                "OOF session-aware calibration at fixed final horizon"
            ),
            "score_rule": (
                "accept if aggregated genuine probability >= threshold_final"
            ),
            "training_info": train_info,
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        model_path = (
            MODELS_DIR
            / (
                "cybellj_"
                + clean_name(
                    protected_user
                )
                + ".pkl"
            )
        )

        joblib.dump(
            package,
            model_path,
            compress=3,
        )

        row = {
            "protected_user": protected_user,
            "candidate": config[
                "candidate"
            ],
            "window_size": WINDOW_SIZE,
            "overlap_percent": OVERLAP_PERCENT,
            "device_context": "mouse",
            "horizon_s": int(
                config[
                    "horizon_s"
                ]
            ),
            "threshold_final": float(
                calibration_row[
                    "threshold_final"
                ]
            ),
            "eer_oof_calibration": float(
                calibration_row[
                    "eer_oof_calibration"
                ]
            ),
            "auc_oof_calibration": float(
                calibration_row[
                    "auc_oof_calibration"
                ]
            ),
            "f1_oof_calibration": float(
                calibration_row[
                    "f1_oof_calibration"
                ]
            ),
            "balanced_accuracy_oof_calibration": float(
                calibration_row[
                    "balanced_accuracy_oof_calibration"
                ]
            ),
            "reference_eer_previous_experiment": float(
                config[
                    "reference_eer"
                ]
            ),
            "n_train_total": train_info[
                "n_train_total"
            ],
            "n_train_genuine": train_info[
                "n_train_genuine"
            ],
            "n_train_impostor": train_info[
                "n_train_impostor"
            ],
            "model_file": str(
                model_path
            ),
        }

        final_rows.append(
            row
        )

        print(
            f"   salvo: {model_path.name}"
        )

        del (
            model,
            package,
        )

        gc.collect()

    final_df = pd.DataFrame(
        final_rows
    )

    final_df.to_csv(
        FINAL_CONFIG_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    write_readme(
        summaries=final_df,
        n_features=len(
            features
        ),
    )

    global_calibration = {
        "eer_mean": float(
            final_df[
                "eer_oof_calibration"
            ].mean()
        ),
        "eer_median": float(
            final_df[
                "eer_oof_calibration"
            ].median()
        ),
        "auc_mean": float(
            final_df[
                "auc_oof_calibration"
            ].mean()
        ),
        "f1_mean": float(
            final_df[
                "f1_oof_calibration"
            ].mean()
        ),
        "balanced_accuracy_mean": float(
            final_df[
                "balanced_accuracy_oof_calibration"
            ].mean()
        ),
    }

    manifest = {
        "project": "CYBELL-J",
        "phase": (
            "13 - final models"
        ),
        "status": (
            "MODELING_FINISHED"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "python": sys.version,
        "dataset": str(
            DATASET_PATH
        ),
        "n_rows_final_configuration": int(
            len(
                df
            )
        ),
        "n_features": int(
            len(
                features
            )
        ),
        "window_size": WINDOW_SIZE,
        "overlap_percent": OVERLAP_PERCENT,
        "device_context": "mouse",
        "n_folds_calibration": N_FOLDS,
        "split_rule": (
            "session_id; no session overlap between train and OOF test"
        ),
        "impostor_to_genuine_train_ratio": (
            IMPOSTOR_TO_GENUINE_TRAIN_RATIO
        ),
        "model_selection_note": (
            "Model families and temporal horizons were frozen before "
            "this script. Phase 13 performs only calibration and final refit."
        ),
        "pycaret_note": (
            "PyCaret was used during experimental model screening/selection. "
            "Final artifacts use underlying scikit-learn/XGBoost estimators "
            "for portability."
        ),
        "lstm_note": (
            "LSTM/TensorFlow was evaluated in Phase 12 but was not adopted "
            "as the standard final pipeline because the classical pipeline "
            "performed better overall and in 5 of 6 profiles."
        ),
        "global_oof_calibration": global_calibration,
        "profiles": final_rows,
        "files": {
            "models_dir": str(
                MODELS_DIR
            ),
            "final_config": str(
                FINAL_CONFIG_PATH
            ),
            "thresholds": str(
                THRESHOLDS_PATH
            ),
            "features": str(
                FEATURES_PATH
            ),
            "calibration_predictions": str(
                CALIBRATION_PRED_PATH
            ),
            "calibration_summary": str(
                CALIBRATION_SUMMARY_PATH
            ),
            "readme": str(
                README_PATH
            ),
        },
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=4,
            default=str,
        )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "3) CONFIGURAÇÃO FINAL"
    )

    print(
        "="
        * 112
    )

    print(
        final_df[
            [
                "protected_user",
                "candidate",
                "horizon_s",
                "threshold_final",
                "eer_oof_calibration",
                "auc_oof_calibration",
                "f1_oof_calibration",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRESULTADO GLOBAL DA CALIBRAÇÃO FINAL:"
    )

    print(
        json.dumps(
            global_calibration,
            ensure_ascii=False,
            indent=4,
        )
    )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "CYBELL-J — MODELAGEM FINALIZADA"
    )

    print(
        "="
        * 112
    )

    print(
        f"\nModelos finais:\n{MODELS_DIR}"
    )

    print(
        f"\nConfiguração final:\n{FINAL_CONFIG_PATH}"
    )

    print(
        f"\nThresholds:\n{THRESHOLDS_PATH}"
    )

    print(
        f"\nFeatures:\n{FEATURES_PATH}"
    )

    print(
        f"\nManifesto final:\n{MANIFEST_PATH}"
    )

    print(
        f"\nREADME de integração:\n{README_PATH}"
    )

    print(
        "\nFIM DA MODELAGEM."
    )


if __name__ == "__main__":
    main()