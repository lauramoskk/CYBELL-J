"""
06_treinar_dispositivo.py

CYBELL-J — Fase 2
Treinamento do classificador de contexto de dispositivo:
    MOUSE x TRACKPAD

Escopo metodológico desta primeira versão:
- Usa SOMENTE as janelas de alta confiança do arquivo
  dataset_device_high_confidence.csv.
- Usa SOMENTE a usuária Bianca, porque é a única participante com
  quantidade relevante de exemplos nos dois contextos.
- O alvo device_context foi produzido pela heurística do front-end.
  Portanto, este experimento é uma prova de conceito intrausuária e
  NÃO deve ser descrito como validação universal de hardware.
- Exclui TODAS as features de wheel/scroll do treinamento principal,
  porque o front-end utilizou scroll para formar o próprio rótulo.
- Exclui todas as colunas que entregariam direta ou indiretamente o
  rótulo de dispositivo.
- Nunca mistura janelas da mesma sessão entre treino e teste.
- Compara Random Forest e XGBoost.
- Avalia as 12 combinações:
      25, 30, 50, 100 eventos
      x
      25%, 50%, 75% de sobreposição
- Calcula F1, EER, ROC-AUC, precisão, recall, acurácia e
  balanced accuracy.
- Registra também duração/latência das janelas e cobertura em
  5, 10, 15 e 20 segundos.
- Salva o melhor modelo e os logs brutos.

Entrada:
    data/mouse_dataset/dataset_device_high_confidence.csv

Saídas:
    data/mouse_training_device/
        resultados_dispositivo_folds.csv
        resultados_dispositivo_resumo.csv
        predicoes_dispositivo.csv
        melhor_modelo_dispositivo.joblib
        melhor_modelo_features.json
        manifesto_treinamento_dispositivo.json
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    raise SystemExit(
        "\nERRO: scikit-learn/joblib não estão instalados.\n"
        "Execute:\n"
        "    pip install scikit-learn joblib\n"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "\nERRO: xgboost não está instalado.\n"
        "Execute:\n"
        "    pip install xgboost\n"
        "Depois rode este script novamente.\n"
    ) from exc


# ============================================================
# CONFIGURAÇÃO
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
    / "mouse_training_device"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FOLDS_PATH = OUTPUT_DIR / "resultados_dispositivo_folds.csv"
SUMMARY_PATH = OUTPUT_DIR / "resultados_dispositivo_resumo.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predicoes_dispositivo.csv"
MODEL_PATH = OUTPUT_DIR / "melhor_modelo_dispositivo.joblib"
FEATURES_PATH = OUTPUT_DIR / "melhor_modelo_features.json"
MANIFEST_PATH = OUTPUT_DIR / "manifesto_treinamento_dispositivo.json"

TARGET_USER = "bianca"
TARGET_COLUMN = "device_context"

WINDOW_SIZES = [25, 30, 50, 100]
OVERLAPS = [25, 50, 75]

# 3 repetições de holdout por grupos/sessões.
# Com apenas 6 sessões da Bianca, aumentar muito esse número não cria
# informação independente nova; apenas repete combinações de sessões.
N_GROUP_HOLDOUTS = 3

RANDOM_STATE = 42

# Classe positiva usada no cálculo da ROC/EER.
POSITIVE_LABEL = "trackpad"

# O modelo principal NÃO usa scroll/wheel.
FEATURE_SET_NAME = "kinematic_click_no_wheel"

# Se True, reduz mensagens internas do XGBoost.
XGB_VERBOSITY = 0


# ============================================================
# COLUNAS QUE NUNCA PODEM VIRAR FEATURES
# ============================================================

EXPLICIT_METADATA_COLUMNS = {
    "target_user",
    "user",
    "session_id",
    "window_id",
    "window_size",
    "requested_overlap_pct",
    "step_events",
    "effective_overlap_pct",
    "window_start_ms",
    "window_end_ms",
    "n_events",

    # Rótulo e derivados — vazamento direto
    "device_mouse_pct",
    "device_trackpad_pct",
    "device_context",
    "device_confidence",
    "device_high_confidence",
    "device_label_source",
    "target_user_device",
}

# Estas features também são excluídas no experimento principal.
# Motivo: envolvem diretamente wheel/scroll ou usam todos os eventos,
# incluindo wheel, e poderiam reproduzir parcialmente a heurística.
EXCLUDED_PRIMARY_FEATURE_PATTERNS = (
    "wheel",
    "event_dt_s",
)

EXCLUDED_PRIMARY_FEATURE_NAMES = {
    "event_rate_hz",
    # duration_s fica fora das features: será usada apenas como latência.
    "duration_s",
}


# ============================================================
# FUNÇÕES DE MÉTRICAS
# ============================================================

def compute_eer(y_true_binary: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Calcula EER aproximado a partir da curva ROC.

    EER ocorre no ponto em que:
        FAR (FPR) ~= FRR (1 - TPR)

    Retorna:
        eer, threshold
    """
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true_binary)) < 2:
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(
        y_true_binary,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    idx = int(np.nanargmin(np.abs(fpr - fnr)))

    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])

    return eer, threshold


def safe_roc_auc(y_true_binary, scores) -> float:
    if len(np.unique(y_true_binary)) < 2:
        return float("nan")
    return float(
        roc_auc_score(
            y_true_binary,
            scores,
        )
    )


def latency_metrics(duration_s: pd.Series) -> dict:
    duration = pd.to_numeric(
        duration_s,
        errors="coerce",
    ).dropna()

    if len(duration) == 0:
        return {
            "latency_mean_s": float("nan"),
            "latency_median_s": float("nan"),
            "latency_p90_s": float("nan"),
            "latency_p95_s": float("nan"),
            "pct_within_5s": float("nan"),
            "pct_within_10s": float("nan"),
            "pct_within_15s": float("nan"),
            "pct_within_20s": float("nan"),
        }

    return {
        "latency_mean_s": float(duration.mean()),
        "latency_median_s": float(duration.median()),
        "latency_p90_s": float(duration.quantile(0.90)),
        "latency_p95_s": float(duration.quantile(0.95)),
        "pct_within_5s": float((duration <= 5).mean() * 100),
        "pct_within_10s": float((duration <= 10).mean() * 100),
        "pct_within_15s": float((duration <= 15).mean() * 100),
        "pct_within_20s": float((duration <= 20).mean() * 100),
    }


# ============================================================
# FEATURES
# ============================================================

def select_primary_features(df: pd.DataFrame) -> list[str]:
    """
    Seleciona apenas features numéricas defensáveis para o primeiro
    classificador mouse x trackpad.

    Exclui:
    - metadados/identificadores;
    - rótulo e derivados;
    - wheel/scroll;
    - métricas de todos os eventos que incluem wheel;
    - duration_s como feature (fica apenas como medida de latência).
    """

    candidate_columns = []

    for column in df.columns:

        if column in EXPLICIT_METADATA_COLUMNS:
            continue

        if column in EXCLUDED_PRIMARY_FEATURE_NAMES:
            continue

        lower = column.lower()

        if any(
            pattern in lower
            for pattern in EXCLUDED_PRIMARY_FEATURE_PATTERNS
        ):
            continue

        # Apenas colunas numéricas.
        if pd.api.types.is_numeric_dtype(df[column]):
            candidate_columns.append(column)

    if not candidate_columns:
        raise RuntimeError(
            "Nenhuma feature numérica foi selecionada."
        )

    return candidate_columns


# ============================================================
# DIVISÃO POR SESSÃO
# ============================================================

def generate_group_holdouts(
    df_config: pd.DataFrame,
    n_splits: int = N_GROUP_HOLDOUTS,
) -> list[dict]:
    """
    Cria divisões determinísticas de treino/teste por session_id.

    As janelas de uma mesma sessão nunca são divididas entre treino e teste.

    Como a Bianca possui poucas sessões, o algoritmo testa combinações de
    2 ou 3 sessões para o conjunto de teste e escolhe divisões que:
      - tenham mouse e trackpad no treino;
      - tenham mouse e trackpad no teste;
      - tenham proporções de classe razoavelmente próximas;
      - mantenham cerca de 1/3 a 1/2 das sessões no teste.
    """

    sessions = sorted(
        df_config["session_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(sessions) < 4:
        raise RuntimeError(
            f"Poucas sessões para holdout por grupo: {len(sessions)}"
        )

    global_trackpad_rate = float(
        (df_config[TARGET_COLUMN] == POSITIVE_LABEL).mean()
    )

    candidates = []

    possible_test_sizes = sorted(
        set(
            [
                max(1, math.floor(len(sessions) / 3)),
                max(2, math.ceil(len(sessions) / 3)),
                max(2, len(sessions) // 2),
            ]
        )
    )

    for test_size in possible_test_sizes:

        if test_size >= len(sessions):
            continue

        for test_groups_tuple in itertools.combinations(
            sessions,
            test_size,
        ):

            test_groups = set(test_groups_tuple)
            train_groups = set(sessions) - test_groups

            train = df_config[
                df_config["session_id"].astype(str).isin(train_groups)
            ]

            test = df_config[
                df_config["session_id"].astype(str).isin(test_groups)
            ]

            if train.empty or test.empty:
                continue

            train_classes = set(
                train[TARGET_COLUMN].dropna().astype(str).unique()
            )

            test_classes = set(
                test[TARGET_COLUMN].dropna().astype(str).unique()
            )

            needed = {"mouse", "trackpad"}

            if not needed.issubset(train_classes):
                continue

            if not needed.issubset(test_classes):
                continue

            train_trackpad_rate = float(
                (train[TARGET_COLUMN] == POSITIVE_LABEL).mean()
            )

            test_trackpad_rate = float(
                (test[TARGET_COLUMN] == POSITIVE_LABEL).mean()
            )

            test_fraction = len(test) / len(df_config)

            # Menor é melhor.
            score = (
                abs(train_trackpad_rate - global_trackpad_rate)
                + abs(test_trackpad_rate - global_trackpad_rate)
                + 0.50 * abs(test_fraction - 0.35)
            )

            candidates.append(
                {
                    "score": score,
                    "test_groups": sorted(test_groups),
                    "train_groups": sorted(train_groups),
                    "train_trackpad_rate": train_trackpad_rate,
                    "test_trackpad_rate": test_trackpad_rate,
                    "test_fraction": test_fraction,
                }
            )

    if not candidates:
        raise RuntimeError(
            "Não foi possível criar um holdout por sessão "
            "com as duas classes em treino e teste."
        )

    candidates = sorted(
        candidates,
        key=lambda item: (
            item["score"],
            item["test_groups"],
        ),
    )

    selected = []
    seen = set()

    for candidate in candidates:

        key = tuple(candidate["test_groups"])

        if key in seen:
            continue

        seen.add(key)
        selected.append(candidate)

        if len(selected) >= n_splits:
            break

    return selected


# ============================================================
# MODELOS
# ============================================================

def build_random_forest() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost(
    y_train_binary: np.ndarray,
) -> Pipeline:

    positives = int(np.sum(y_train_binary == 1))
    negatives = int(np.sum(y_train_binary == 0))

    if positives > 0:
        scale_pos_weight = negatives / positives
    else:
        scale_pos_weight = 1.0

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=XGB_VERBOSITY,
                ),
            ),
        ]
    )


# ============================================================
# TREINAMENTO DE UM FOLD
# ============================================================

def train_evaluate_one_fold(
    model_name: str,
    model,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    window_size: int,
    overlap: int,
    fold_number: int,
    train_groups: list[str],
    test_groups: list[str],
) -> tuple[dict, pd.DataFrame]:

    X_train = df_train[features].copy()
    X_test = df_test[features].copy()

    y_train = (
        df_train[TARGET_COLUMN]
        .astype(str)
        .eq(POSITIVE_LABEL)
        .astype(int)
        .to_numpy()
    )

    y_test = (
        df_test[TARGET_COLUMN]
        .astype(str)
        .eq(POSITIVE_LABEL)
        .astype(int)
        .to_numpy()
    )

    start_fit = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    fit_seconds = time.perf_counter() - start_fit

    start_predict = time.perf_counter()

    proba = model.predict_proba(X_test)[:, 1]

    prediction_seconds = time.perf_counter() - start_predict

    pred = (proba >= 0.5).astype(int)

    eer, eer_threshold = compute_eer(
        y_test,
        proba,
    )

    f1 = float(
        f1_score(
            y_test,
            pred,
            pos_label=1,
            zero_division=0,
        )
    )

    precision = float(
        precision_score(
            y_test,
            pred,
            pos_label=1,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_test,
            pred,
            pos_label=1,
            zero_division=0,
        )
    )

    accuracy = float(
        accuracy_score(
            y_test,
            pred,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y_test,
            pred,
        )
    )

    roc_auc = safe_roc_auc(
        y_test,
        proba,
    )

    latency = latency_metrics(
        df_test["duration_s"]
    )

    result = {
        "feature_set": FEATURE_SET_NAME,
        "model": model_name,
        "window_size": int(window_size),
        "overlap_pct": int(overlap),
        "fold": int(fold_number),

        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_features": int(len(features)),

        "train_mouse": int(
            (df_train[TARGET_COLUMN] == "mouse").sum()
        ),
        "train_trackpad": int(
            (df_train[TARGET_COLUMN] == "trackpad").sum()
        ),
        "test_mouse": int(
            (df_test[TARGET_COLUMN] == "mouse").sum()
        ),
        "test_trackpad": int(
            (df_test[TARGET_COLUMN] == "trackpad").sum()
        ),

        "train_sessions": "|".join(train_groups),
        "test_sessions": "|".join(test_groups),

        "f1": f1,
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,

        "fit_seconds": float(fit_seconds),
        "predict_seconds_total": float(prediction_seconds),
        "predict_ms_per_window": float(
            prediction_seconds
            / max(len(df_test), 1)
            * 1000.0
        ),

        **latency,
    }

    pred_df = pd.DataFrame(
        {
            "model": model_name,
            "feature_set": FEATURE_SET_NAME,
            "window_size": window_size,
            "overlap_pct": overlap,
            "fold": fold_number,
            "session_id": df_test["session_id"].astype(str).to_numpy(),
            "window_id": df_test["window_id"].astype(str).to_numpy(),
            "y_true": y_test,
            "y_true_label": np.where(
                y_test == 1,
                "trackpad",
                "mouse",
            ),
            "proba_trackpad": proba,
            "y_pred": pred,
            "y_pred_label": np.where(
                pred == 1,
                "trackpad",
                "mouse",
            ),
            "duration_s": pd.to_numeric(
                df_test["duration_s"],
                errors="coerce",
            ).to_numpy(),
        }
    )

    return result, pred_df


# ============================================================
# RESUMO DOS FOLDS
# ============================================================

def summarize_results(
    folds_df: pd.DataFrame,
) -> pd.DataFrame:

    metric_columns = [
        "f1",
        "eer",
        "roc_auc",
        "precision",
        "recall",
        "accuracy",
        "balanced_accuracy",
        "latency_mean_s",
        "latency_median_s",
        "latency_p90_s",
        "latency_p95_s",
        "pct_within_5s",
        "pct_within_10s",
        "pct_within_15s",
        "pct_within_20s",
        "fit_seconds",
        "predict_ms_per_window",
    ]

    rows = []

    grouped = folds_df.groupby(
        [
            "feature_set",
            "model",
            "window_size",
            "overlap_pct",
        ],
        dropna=False,
    )

    for keys, group in grouped:

        feature_set, model, window_size, overlap_pct = keys

        row = {
            "feature_set": feature_set,
            "model": model,
            "window_size": int(window_size),
            "overlap_pct": int(overlap_pct),
            "n_folds": int(len(group)),
        }

        for metric in metric_columns:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Regra de ordenação alinhada ao guia:
    # 1) menor EER
    # 2) maior F1
    # 3) menor p90 de latência
    summary = summary.sort_values(
        [
            "eer_mean",
            "f1_mean",
            "latency_p90_s_mean",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1
        ),
    )

    return summary


# ============================================================
# MODELO FINAL
# ============================================================

def train_final_best_model(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    features: list[str],
) -> tuple[object, dict]:

    best = summary_df.iloc[0]

    model_name = str(best["model"])
    window_size = int(best["window_size"])
    overlap_pct = int(best["overlap_pct"])

    subset = df[
        (df["window_size"] == window_size)
        & (
            df["requested_overlap_pct"]
            == overlap_pct
        )
    ].copy()

    X = subset[features].copy()

    y = (
        subset[TARGET_COLUMN]
        .astype(str)
        .eq(POSITIVE_LABEL)
        .astype(int)
        .to_numpy()
    )

    if model_name == "RandomForest":
        model = build_random_forest()
    elif model_name == "XGBoost":
        model = build_xgboost(y)
    else:
        raise RuntimeError(
            f"Modelo desconhecido: {model_name}"
        )

    model.fit(
        X,
        y,
    )

    info = {
        "model": model_name,
        "window_size": window_size,
        "overlap_pct": overlap_pct,
        "feature_set": FEATURE_SET_NAME,
        "n_features": len(features),
        "n_training_windows": len(subset),
        "mouse_windows": int(
            (subset[TARGET_COLUMN] == "mouse").sum()
        ),
        "trackpad_windows": int(
            (subset[TARGET_COLUMN] == "trackpad").sum()
        ),
        "selection_eer_mean": float(best["eer_mean"]),
        "selection_f1_mean": float(best["f1_mean"]),
        "selection_latency_p90_s_mean": float(
            best["latency_p90_s_mean"]
        ),
    }

    return model, info


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 82)
    print("CYBELL-J — TREINAMENTO DE DISPOSITIVO: MOUSE x TRACKPAD")
    print("=" * 82)

    print(
        "\nIMPORTANTE:"
        "\n- Experimento intrausuária (Bianca)."
        "\n- O alvo mouse/trackpad deriva da heurística do front-end."
        "\n- Features de wheel/scroll NÃO entram no modelo principal."
        "\n- Sessões são mantidas inteiras em treino ou teste."
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset não encontrado:\n{DATASET_PATH}"
        )

    print("\n1) Carregando dataset...")

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False,
    )

    required = {
        "user",
        "session_id",
        "window_id",
        "window_size",
        "requested_overlap_pct",
        "duration_s",
        TARGET_COLUMN,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Colunas obrigatórias ausentes: {missing}"
        )

    df = df[
        df["user"]
        .astype(str)
        .str.lower()
        .eq(TARGET_USER.lower())
    ].copy()

    df = df[
        df[TARGET_COLUMN].isin(
            ["mouse", "trackpad"]
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            "Nenhuma janela da Bianca com mouse/trackpad foi encontrada."
        )

    print(
        f"   ✅ Janelas da Bianca: {len(df):,}"
        .replace(",", ".")
    )

    print(
        f"   Mouse: "
        f"{int((df[TARGET_COLUMN] == 'mouse').sum()):,}"
        .replace(",", ".")
    )

    print(
        f"   Trackpad: "
        f"{int((df[TARGET_COLUMN] == 'trackpad').sum()):,}"
        .replace(",", ".")
    )

    print(
        f"   Sessões: {df['session_id'].nunique()}"
    )

    features = select_primary_features(
        df
    )

    print(
        f"\n2) Features selecionadas: {len(features)}"
    )

    print(
        "   Wheel/scroll foi excluído do conjunto principal."
    )

    print("\n   Features:")
    for feature in features:
        print(f"   - {feature}")

    all_fold_results = []
    all_predictions = []

    print("\n" + "=" * 82)
    print("3) INICIANDO OS 12 EXPERIMENTOS")
    print("=" * 82)

    total_configs = len(WINDOW_SIZES) * len(OVERLAPS)
    config_number = 0

    for window_size in WINDOW_SIZES:

        for overlap in OVERLAPS:

            config_number += 1

            config = df[
                (df["window_size"] == window_size)
                & (
                    df["requested_overlap_pct"]
                    == overlap
                )
            ].copy()

            print(
                f"\n[{config_number}/{total_configs}] "
                f"{window_size} eventos | overlap {overlap}%"
            )

            print(
                f"   Janelas: {len(config):,}"
                .replace(",", ".")
            )

            print(
                f"   Mouse: "
                f"{int((config[TARGET_COLUMN] == 'mouse').sum()):,}"
                .replace(",", ".")
            )

            print(
                f"   Trackpad: "
                f"{int((config[TARGET_COLUMN] == 'trackpad').sum()):,}"
                .replace(",", ".")
            )

            print(
                f"   Sessões: {config['session_id'].nunique()}"
            )

            if len(config) == 0:
                print("   ⚠️ Sem dados. Pulando.")
                continue

            holdouts = generate_group_holdouts(
                config,
                n_splits=N_GROUP_HOLDOUTS,
            )

            for fold_number, split in enumerate(
                holdouts,
                start=1,
            ):

                train_groups = split["train_groups"]
                test_groups = split["test_groups"]

                train = config[
                    config["session_id"]
                    .astype(str)
                    .isin(train_groups)
                ].copy()

                test = config[
                    config["session_id"]
                    .astype(str)
                    .isin(test_groups)
                ].copy()

                y_train_binary = (
                    train[TARGET_COLUMN]
                    .eq(POSITIVE_LABEL)
                    .astype(int)
                    .to_numpy()
                )

                models = {
                    "RandomForest": build_random_forest(),
                    "XGBoost": build_xgboost(
                        y_train_binary
                    ),
                }

                print(
                    f"   Fold {fold_number}: "
                    f"teste={','.join(test_groups)}"
                )

                for model_name, model in models.items():

                    result, pred_df = train_evaluate_one_fold(
                        model_name=model_name,
                        model=model,
                        df_train=train,
                        df_test=test,
                        features=features,
                        window_size=window_size,
                        overlap=overlap,
                        fold_number=fold_number,
                        train_groups=train_groups,
                        test_groups=test_groups,
                    )

                    all_fold_results.append(
                        result
                    )

                    all_predictions.append(
                        pred_df
                    )

                    print(
                        f"      {model_name:<13} "
                        f"F1={result['f1']:.4f} | "
                        f"EER={result['eer']:.4f} | "
                        f"AUC={result['roc_auc']:.4f}"
                    )

    if not all_fold_results:
        raise RuntimeError(
            "Nenhum treinamento foi concluído."
        )

    folds_df = pd.DataFrame(
        all_fold_results
    )

    folds_df.to_csv(
        FOLDS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = summarize_results(
        folds_df
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MELHOR CONFIGURAÇÃO
    # ========================================================

    print("\n" + "=" * 82)
    print("4) MELHOR CONFIGURAÇÃO")
    print("=" * 82)

    best = summary_df.iloc[0]

    print(
        f"\nModelo: {best['model']}"
    )
    print(
        f"Janela: {int(best['window_size'])} eventos"
    )
    print(
        f"Overlap: {int(best['overlap_pct'])}%"
    )
    print(
        f"F1 médio: {best['f1_mean']:.4f}"
    )
    print(
        f"EER médio: {best['eer_mean']:.4f}"
    )
    print(
        f"ROC-AUC médio: {best['roc_auc_mean']:.4f}"
    )
    print(
        f"Latência p90 média: "
        f"{best['latency_p90_s_mean']:.3f} s"
    )
    print(
        f"Janelas <= 5 s: "
        f"{best['pct_within_5s_mean']:.2f}%"
    )
    print(
        f"Janelas <= 10 s: "
        f"{best['pct_within_10s_mean']:.2f}%"
    )
    print(
        f"Janelas <= 15 s: "
        f"{best['pct_within_15s_mean']:.2f}%"
    )
    print(
        f"Janelas <= 20 s: "
        f"{best['pct_within_20s_mean']:.2f}%"
    )

    print(
        "\n5) Treinando o melhor modelo em todas "
        "as janelas da configuração vencedora..."
    )

    final_model, final_info = train_final_best_model(
        df,
        summary_df,
        features,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    features_payload = {
        "feature_set": FEATURE_SET_NAME,
        "features": features,
        "target": TARGET_COLUMN,
        "positive_class": POSITIVE_LABEL,
        "model_info": final_info,
    }

    with FEATURES_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            features_payload,
            f,
            ensure_ascii=False,
            indent=4,
        )

    manifest = {
        "projeto": "CYBELL-J",
        "fase": "Fase 2 - Identificação do Tipo de Dispositivo de Apontamento",
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": str(DATASET_PATH.name),
        "target_user": TARGET_USER,
        "target": TARGET_COLUMN,
        "positive_class": POSITIVE_LABEL,
        "feature_set": FEATURE_SET_NAME,
        "window_sizes": WINDOW_SIZES,
        "overlaps_pct": OVERLAPS,
        "n_group_holdouts": N_GROUP_HOLDOUTS,
        "split_policy": (
            "Holdout por session_id; uma sessão nunca aparece "
            "simultaneamente em treino e teste."
        ),
        "models": [
            "RandomForest",
            "XGBoost",
        ],
        "best_model": final_info,
        "outputs": {
            "fold_results": str(FOLDS_PATH.name),
            "summary": str(SUMMARY_PATH.name),
            "predictions": str(PREDICTIONS_PATH.name),
            "model": str(MODEL_PATH.name),
            "features": str(FEATURES_PATH.name),
        },
        "methodological_notes": [
            "O experimento de dispositivo usa apenas a usuária Bianca, pois é a única com volume relevante de mouse e trackpad.",
            "O rótulo device_context deriva da heurística do front-end e não constitui ground truth físico independente.",
            "Features de wheel/scroll foram excluídas do experimento principal para reduzir circularidade, pois o front-end usou scroll na geração do rótulo.",
            "Todas as janelas de uma mesma session_id permanecem juntas em treino ou teste para reduzir vazamento causado por sobreposição.",
            "A classe positiva para ROC/EER é trackpad.",
            "O modelo final salvo é treinado em todas as janelas da configuração escolhida após a comparação por holdout de sessões.",
            "Este resultado deve ser descrito como prova de conceito intrausuária; não como detector universal de mouse/trackpad.",
        ],
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
        )

    print("\n" + "=" * 82)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 82)

    print(
        f"\nResultados por fold:\n{FOLDS_PATH}"
    )

    print(
        f"\nResumo das configurações:\n{SUMMARY_PATH}"
    )

    print(
        f"\nPredições brutas:\n{PREDICTIONS_PATH}"
    )

    print(
        f"\nModelo salvo:\n{MODEL_PATH}"
    )

    print(
        f"\nFeatures do modelo:\n{FEATURES_PATH}"
    )

    print(
        f"\nManifesto:\n{MANIFEST_PATH}"
    )

    print(
        "\n✅ Próxima etapa após revisar estes resultados:"
        "\n   autenticação biométrica — perfil único versus "
        "subperfil condicionado ao dispositivo."
    )


if __name__ == "__main__":
    main()