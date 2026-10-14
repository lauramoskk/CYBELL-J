"""
07_pycaret_dispositivo.py

CYBELL-J — PyCaret 4.0
Busca de modelos melhores para MOUSE x TRACKPAD.

Objetivo desta etapa:
1) Manter a mesma configuração vencedora do baseline:
      25 eventos / 75% overlap
2) Usar as mesmas features principais SEM wheel/scroll.
3) Usar PyCaret 4 para comparar vários classificadores.
4) Preservar validação por SESSÃO com um cross-validator próprio.
5) Selecionar os 5 melhores por AUC.
6) Avaliar os 5 com os mesmos holdouts de sessão usados no baseline.
7) Tunar os 3 melhores candidatos com PyCaret.
8) Reavaliar F1, Balanced Accuracy, AUC e EER.
9) Salvar o melhor pipeline e todos os logs.

IMPORTANTE
----------
O rótulo device_context continua sendo derivado da heurística do front-end.
Este experimento continua sendo uma prova de conceito intrausuária (Bianca),
e não uma validação universal de hardware.

Entrada:
    data/mouse_dataset/dataset_device_high_confidence.csv

Saídas:
    data/mouse_training_device_pycaret/
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pycaret

try:
    from pycaret.tasks import ClassificationExperiment
except ImportError:
    # fallback de compatibilidade
    from pycaret.classification import ClassificationExperiment

from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


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
    / "mouse_training_device_pycaret"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEADERBOARD_PATH = OUTPUT_DIR / "pycaret_leaderboard.csv"
OOF_BASE_PATH = OUTPUT_DIR / "pycaret_avaliacao_top5.csv"
OOF_TUNED_PATH = OUTPUT_DIR / "pycaret_avaliacao_tunados.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "pycaret_predicoes_oof.csv"
TUNING_PATH = OUTPUT_DIR / "pycaret_tuning_parametros.json"
MODEL_PATH = OUTPUT_DIR / "melhor_modelo_pycaret.joblib"
FEATURES_PATH = OUTPUT_DIR / "melhor_modelo_pycaret_features.json"
MANIFEST_PATH = OUTPUT_DIR / "manifesto_pycaret_dispositivo.json"

TARGET_USER = "bianca"
TARGET_COLUMN = "device_context"
POSITIVE_LABEL = "trackpad"

# Nesta primeira rodada PyCaret, usamos a melhor configuração encontrada
# no baseline manual.
WINDOW_SIZE = 25
OVERLAP = 75

RANDOM_STATE = 42

# PyCaret compara vários modelos; avaliamos externamente os 5 primeiros
# e tunamos os 3 candidatos mais fortes.
N_SELECT = 5
N_TUNE = 3
N_TUNE_ITER = 15

# PyCaret precisa criar um pequeno holdout interno.
# A avaliação científica final abaixo NÃO usa esse holdout aleatório:
# ela usa os folds por sessão.
PYCARET_TRAIN_SIZE = 0.95

FEATURE_SET_NAME = "kinematic_click_no_wheel"


# ============================================================
# COLUNAS EXCLUÍDAS
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

    # Rótulo e derivados: vazamento direto
    "device_mouse_pct",
    "device_trackpad_pct",
    "device_context",
    "device_confidence",
    "device_high_confidence",
    "device_label_source",
    "target_user_device",
}

EXCLUDED_PRIMARY_FEATURE_PATTERNS = (
    "wheel",
    "event_dt_s",
)

EXCLUDED_PRIMARY_FEATURE_NAMES = {
    "event_rate_hz",
    "duration_s",
}


# ============================================================
# UTILIDADES
# ============================================================

def select_primary_features(df: pd.DataFrame) -> list[str]:
    features = []

    for column in df.columns:
        if column in EXPLICIT_METADATA_COLUMNS:
            continue

        if column in EXCLUDED_PRIMARY_FEATURE_NAMES:
            continue

        lower = column.lower()

        if any(pattern in lower for pattern in EXCLUDED_PRIMARY_FEATURE_PATTERNS):
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            features.append(column)

    if not features:
        raise RuntimeError("Nenhuma feature numérica foi selecionada.")

    return features


def compute_eer(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
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


def get_continuous_score(model, X) -> np.ndarray:
    """
    Retorna score contínuo para ROC/EER.
    Prioridade:
      1) predict_proba da classe 1
      2) decision_function
    """

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))

        if proba.ndim == 2 and proba.shape[1] >= 2:
            classes = list(getattr(model, "classes_", [0, 1]))

            if 1 in classes:
                positive_idx = classes.index(1)
            else:
                positive_idx = 1

            return proba[:, positive_idx].astype(float)

    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X))

        if score.ndim == 2:
            classes = list(getattr(model, "classes_", range(score.shape[1])))
            positive_idx = classes.index(1) if 1 in classes else 1
            score = score[:, positive_idx]

        return score.astype(float)

    # Último recurso: predição binária.
    return np.asarray(model.predict(X), dtype=float)


def latency_metrics(duration_s: pd.Series) -> dict:
    duration = pd.to_numeric(duration_s, errors="coerce").dropna()

    if len(duration) == 0:
        return {
            "latency_median_s": float("nan"),
            "latency_p90_s": float("nan"),
            "pct_within_5s": float("nan"),
            "pct_within_10s": float("nan"),
            "pct_within_15s": float("nan"),
            "pct_within_20s": float("nan"),
        }

    return {
        "latency_median_s": float(duration.median()),
        "latency_p90_s": float(duration.quantile(0.90)),
        "pct_within_5s": float((duration <= 5).mean() * 100),
        "pct_within_10s": float((duration <= 10).mean() * 100),
        "pct_within_15s": float((duration <= 15).mean() * 100),
        "pct_within_20s": float((duration <= 20).mean() * 100),
    }


# ============================================================
# FOLDS POR SESSÃO
# ============================================================

def generate_group_holdouts(
    df_config: pd.DataFrame,
    n_splits: int = 3,
) -> list[dict]:
    """
    Replica a lógica metodológica do script 06:
    procura divisões de sessões em que mouse e trackpad existam
    tanto no treino quanto no teste.
    """

    import itertools

    sessions = sorted(
        df_config["session_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(sessions) < 4:
        raise RuntimeError(
            f"Poucas sessões para validação por grupo: {len(sessions)}"
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

        for test_tuple in itertools.combinations(sessions, test_size):
            test_groups = set(test_tuple)
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

            train_rate = float(
                (train[TARGET_COLUMN] == POSITIVE_LABEL).mean()
            )

            test_rate = float(
                (test[TARGET_COLUMN] == POSITIVE_LABEL).mean()
            )

            test_fraction = len(test) / len(df_config)

            score = (
                abs(train_rate - global_trackpad_rate)
                + abs(test_rate - global_trackpad_rate)
                + 0.50 * abs(test_fraction - 0.35)
            )

            candidates.append(
                {
                    "score": score,
                    "train_groups": sorted(train_groups),
                    "test_groups": sorted(test_groups),
                }
            )

    if not candidates:
        raise RuntimeError(
            "Não foi possível construir folds por sessão com as duas classes."
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


class SessionPredefinedCV(BaseCrossValidator):
    """
    Cross-validator passado ao PyCaret.

    O PyCaret 4 chama:
        cv.split(X, y)

    Esta classe recupera a session_id usando o índice original de cada
    janela e garante que uma sessão inteira fique em um lado do fold.
    """

    def __init__(
        self,
        index_to_session: dict,
        fold_definitions: list[dict],
    ):
        self.index_to_session = index_to_session
        self.fold_definitions = fold_definitions

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.fold_definitions)

    def split(self, X, y=None, groups=None):
        if not hasattr(X, "index"):
            raise RuntimeError(
                "SessionPredefinedCV exige DataFrame com índice preservado."
            )

        session_values = np.array(
            [
                self.index_to_session.get(idx)
                for idx in X.index
            ],
            dtype=object,
        )

        for fold in self.fold_definitions:
            test_sessions = set(fold["test_groups"])

            test_mask = np.array(
                [s in test_sessions for s in session_values],
                dtype=bool,
            )

            train_idx = np.flatnonzero(~test_mask)
            test_idx = np.flatnonzero(test_mask)

            if len(train_idx) == 0 or len(test_idx) == 0:
                raise RuntimeError(
                    "Fold por sessão ficou vazio após o split interno do PyCaret."
                )

            if y is not None:
                y_arr = np.asarray(y)

                if len(np.unique(y_arr[train_idx])) < 2:
                    raise RuntimeError(
                        "Treino de um fold ficou com apenas uma classe."
                    )

                if len(np.unique(y_arr[test_idx])) < 2:
                    raise RuntimeError(
                        "Teste de um fold ficou com apenas uma classe."
                    )

            yield train_idx, test_idx


# ============================================================
# AVALIAÇÃO EXTERNA CONTROLADA
# ============================================================

def evaluate_pipeline_on_session_folds(
    pipeline,
    model_name: str,
    stage: str,
    df_config: pd.DataFrame,
    features: list[str],
    fold_definitions: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rows = []
    prediction_frames = []

    for fold_number, fold in enumerate(fold_definitions, start=1):
        train_groups = fold["train_groups"]
        test_groups = fold["test_groups"]

        train = df_config[
            df_config["session_id"].astype(str).isin(train_groups)
        ].copy()

        test = df_config[
            df_config["session_id"].astype(str).isin(test_groups)
        ].copy()

        X_train = train[features]
        X_test = test[features]

        y_train = (
            train[TARGET_COLUMN]
            .astype(str)
            .eq(POSITIVE_LABEL)
            .astype(int)
            .to_numpy()
        )

        y_test = (
            test[TARGET_COLUMN]
            .astype(str)
            .eq(POSITIVE_LABEL)
            .astype(int)
            .to_numpy()
        )

        model = clone(pipeline)

        start_fit = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start_fit

        start_predict = time.perf_counter()

        pred = np.asarray(model.predict(X_test)).astype(int)
        score = get_continuous_score(model, X_test)

        predict_seconds = time.perf_counter() - start_predict

        eer, eer_threshold = compute_eer(
            y_test,
            score,
        )

        auc = float(
            roc_auc_score(
                y_test,
                score,
            )
        )

        latency = latency_metrics(
            test["duration_s"]
        )

        row = {
            "stage": stage,
            "model": model_name,
            "fold": fold_number,
            "train_sessions": "|".join(train_groups),
            "test_sessions": "|".join(test_groups),
            "n_train": len(train),
            "n_test": len(test),
            "train_mouse": int(
                (train[TARGET_COLUMN] == "mouse").sum()
            ),
            "train_trackpad": int(
                (train[TARGET_COLUMN] == "trackpad").sum()
            ),
            "test_mouse": int(
                (test[TARGET_COLUMN] == "mouse").sum()
            ),
            "test_trackpad": int(
                (test[TARGET_COLUMN] == "trackpad").sum()
            ),
            "f1": float(
                f1_score(
                    y_test,
                    pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "roc_auc": auc,
            "accuracy": float(
                accuracy_score(
                    y_test,
                    pred,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_test,
                    pred,
                )
            ),
            "precision": float(
                precision_score(
                    y_test,
                    pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            "fit_seconds": float(fit_seconds),
            "predict_ms_per_window": float(
                predict_seconds
                / max(len(test), 1)
                * 1000
            ),
            **latency,
        }

        rows.append(row)

        pred_frame = pd.DataFrame(
            {
                "stage": stage,
                "model": model_name,
                "fold": fold_number,
                "session_id": test["session_id"].astype(str).to_numpy(),
                "window_id": test["window_id"].astype(str).to_numpy(),
                "y_true": y_test,
                "score_trackpad": score,
                "y_pred": pred,
                "duration_s": pd.to_numeric(
                    test["duration_s"],
                    errors="coerce",
                ).to_numpy(),
            }
        )

        prediction_frames.append(pred_frame)

    return (
        pd.DataFrame(rows),
        pd.concat(
            prediction_frames,
            ignore_index=True,
        ),
    )


def summarize_external_evaluation(
    eval_df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "f1",
        "eer",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "latency_median_s",
        "latency_p90_s",
        "pct_within_5s",
        "pct_within_10s",
        "pct_within_15s",
        "pct_within_20s",
        "fit_seconds",
        "predict_ms_per_window",
    ]

    rows = []

    for (stage, model), group in eval_df.groupby(
        ["stage", "model"],
        dropna=False,
    ):
        row = {
            "stage": stage,
            "model": model,
            "n_folds": len(group),
        }

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        [
            "eer_mean",
            "roc_auc_mean",
            "balanced_accuracy_mean",
            "f1_mean",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(1, len(summary) + 1),
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 88)
    print("CYBELL-J — PYCARET 4 — MOUSE x TRACKPAD")
    print("=" * 88)

    print(f"\nPyCaret detectado: {pycaret.__version__}")

    if not str(pycaret.__version__).startswith("4."):
        raise RuntimeError(
            "Este script foi preparado para PyCaret 4.x."
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado:\n{DATASET_PATH}"
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

    missing = sorted(required - set(df.columns))

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
        (df["window_size"] == WINDOW_SIZE)
        & (
            df["requested_overlap_pct"]
            == OVERLAP
        )
        & (
            df[TARGET_COLUMN].isin(
                ["mouse", "trackpad"]
            )
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            "Nenhuma janela encontrada para "
            f"{WINDOW_SIZE} eventos / {OVERLAP}%."
        )

    print(
        f"   Configuração: {WINDOW_SIZE} eventos | overlap {OVERLAP}%"
    )

    print(
        f"   Janelas: {len(df):,}"
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

    features = select_primary_features(df)

    print(
        f"\n2) Features sem wheel/scroll: {len(features)}"
    )

    # Preservar índice original é essencial para o CV por sessão.
    index_to_session = (
        df["session_id"]
        .astype(str)
        .to_dict()
    )

    fold_definitions = generate_group_holdouts(
        df,
        n_splits=3,
    )

    print("\n3) Folds por sessão que serão usados:")

    for i, fold in enumerate(
        fold_definitions,
        start=1,
    ):
        print(
            f"   Fold {i} | "
            f"teste={','.join(fold['test_groups'])}"
        )

    session_cv = SessionPredefinedCV(
        index_to_session=index_to_session,
        fold_definitions=fold_definitions,
    )

    # Dataset entregue ao PyCaret.
    # session_id NÃO vira feature.
    pycaret_data = df[
        features
        + [TARGET_COLUMN]
    ].copy()

    # Converte target para 0/1 para deixar a avaliação externa
    # exatamente consistente em todos os modelos.
    pycaret_data["target_bin"] = (
        pycaret_data[TARGET_COLUMN]
        .astype(str)
        .eq(POSITIVE_LABEL)
        .astype(int)
    )

    pycaret_data = pycaret_data.drop(
        columns=[TARGET_COLUMN]
    )

    print("\n4) Criando experimento PyCaret...")

    exp = ClassificationExperiment(
        target="target_bin",
        session_id=RANDOM_STATE,
        train_size=PYCARET_TRAIN_SIZE,
        fold=3,
        preprocess=True,
        normalize=False,
        transformation=False,
        remove_outliers=False,
        feature_selection=False,
        n_jobs=-1,
        use_gpu=False,
        verbose=False,
    )

    exp.fit(pycaret_data)

    print("\n5) Comparando modelos disponíveis no PyCaret...")
    print(
        "   Critério de ranking interno: AUC"
        "\n   Validação: folds definidos por session_id"
    )

    compare_result = exp.compare_models(
        fold=session_cv,
        sort="AUC",
        n_select=N_SELECT,
        turbo=True,
        errors="ignore",
        round=4,
        verbose=False,
    )

    leaderboard = compare_result.leaderboard.copy()

    leaderboard.to_csv(
        LEADERBOARD_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== LEADERBOARD PYCARET ===")
    print(
        leaderboard.head(15).to_string(index=False)
    )

    if not compare_result.models:
        raise RuntimeError(
            "PyCaret não conseguiu treinar nenhum modelo."
        )

    top_models = list(compare_result.models)
    ranked_ids = list(compare_result.ranked_ids)

    print(
        f"\n6) Avaliando os {len(top_models)} "
        "melhores em holdouts externos por sessão..."
    )

    base_eval_parts = []
    prediction_parts = []

    top_registry_ids = []

    for rank, pipeline in enumerate(
        top_models,
        start=1,
    ):
        model_id = (
            ranked_ids[rank - 1]
            if rank - 1 < len(ranked_ids)
            else f"modelo_{rank}"
        )

        top_registry_ids.append(model_id)

        model_name = f"{rank:02d}_{model_id}"

        print(
            f"\n   Avaliando {model_name}..."
        )

        eval_df, pred_df = evaluate_pipeline_on_session_folds(
            pipeline=pipeline,
            model_name=model_name,
            stage="pycaret_default",
            df_config=df,
            features=features,
            fold_definitions=fold_definitions,
        )

        base_eval_parts.append(eval_df)
        prediction_parts.append(pred_df)

        tmp_summary = summarize_external_evaluation(
            eval_df
        ).iloc[0]

        print(
            f"      F1={tmp_summary['f1_mean']:.4f} | "
            f"EER={tmp_summary['eer_mean']:.4f} | "
            f"AUC={tmp_summary['roc_auc_mean']:.4f} | "
            f"BalAcc={tmp_summary['balanced_accuracy_mean']:.4f}"
        )

    base_eval_df = pd.concat(
        base_eval_parts,
        ignore_index=True,
    )

    base_summary = summarize_external_evaluation(
        base_eval_df
    )

    base_summary.to_csv(
        OOF_BASE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== TOP 5 — AVALIAÇÃO EXTERNA POR SESSÃO ===")
    print(
        base_summary[
            [
                "rank",
                "model",
                "f1_mean",
                "eer_mean",
                "roc_auc_mean",
                "balanced_accuracy_mean",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # TUNING
    # ========================================================

    print(
        f"\n7) Tunando os {min(N_TUNE, len(top_models))} "
        "melhores candidatos..."
    )

    tuned_eval_parts = []
    tuned_pipelines = {}
    tuning_log = {}

    # Para tunar, priorizamos os modelos que realmente foram melhores
    # na avaliação externa por sessão, não apenas no leaderboard interno.
    best_default_names = (
        base_summary["model"]
        .head(N_TUNE)
        .tolist()
    )

    model_by_name = {}

    for rank, pipeline in enumerate(
        top_models,
        start=1,
    ):
        model_id = (
            ranked_ids[rank - 1]
            if rank - 1 < len(ranked_ids)
            else f"modelo_{rank}"
        )

        model_name = f"{rank:02d}_{model_id}"
        model_by_name[model_name] = pipeline

    for model_name in best_default_names:

        pipeline = model_by_name[model_name]

        print(
            f"\n   Tunando {model_name} "
            f"({N_TUNE_ITER} iterações)..."
        )

        try:
            tune_result = exp.tune_model(
                pipeline,
                fold=session_cv,
                n_iter=N_TUNE_ITER,
                optimize="AUC",
                round=4,
                verbose=False,
            )

            tuned_pipeline = tune_result.pipeline
            tuned_name = f"{model_name}_TUNED"

            tuned_pipelines[tuned_name] = tuned_pipeline

            best_params = getattr(
                tune_result,
                "best_params",
                {},
            )

            tuning_log[tuned_name] = {
                "best_params": best_params,
            }

            eval_df, pred_df = evaluate_pipeline_on_session_folds(
                pipeline=tuned_pipeline,
                model_name=tuned_name,
                stage="pycaret_tuned",
                df_config=df,
                features=features,
                fold_definitions=fold_definitions,
            )

            tuned_eval_parts.append(eval_df)
            prediction_parts.append(pred_df)

            tmp_summary = summarize_external_evaluation(
                eval_df
            ).iloc[0]

            print(
                f"      F1={tmp_summary['f1_mean']:.4f} | "
                f"EER={tmp_summary['eer_mean']:.4f} | "
                f"AUC={tmp_summary['roc_auc_mean']:.4f} | "
                f"BalAcc={tmp_summary['balanced_accuracy_mean']:.4f}"
            )

        except Exception as exc:
            print(
                f"      ⚠️ Tuning falhou para {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )

            tuning_log[f"{model_name}_TUNING_FAILED"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }

    if tuned_eval_parts:
        tuned_eval_df = pd.concat(
            tuned_eval_parts,
            ignore_index=True,
        )

        tuned_summary = summarize_external_evaluation(
            tuned_eval_df
        )

        tuned_summary.to_csv(
            OOF_TUNED_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n=== MODELOS TUNADOS — AVALIAÇÃO EXTERNA ===")

        print(
            tuned_summary[
                [
                    "rank",
                    "model",
                    "f1_mean",
                    "eer_mean",
                    "roc_auc_mean",
                    "balanced_accuracy_mean",
                ]
            ].to_string(index=False)
        )

    else:
        tuned_eval_df = pd.DataFrame()
        tuned_summary = pd.DataFrame()

        pd.DataFrame().to_csv(
            OOF_TUNED_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    with TUNING_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            tuning_log,
            f,
            ensure_ascii=False,
            indent=4,
            default=str,
        )

    # ========================================================
    # ESCOLHA FINAL ENTRE DEFAULT + TUNADOS
    # ========================================================

    print("\n8) Selecionando melhor resultado PyCaret...")

    combined_eval = pd.concat(
        [
            base_eval_df,
            tuned_eval_df,
        ],
        ignore_index=True,
    )

    combined_summary = summarize_external_evaluation(
        combined_eval
    )

    best_row = combined_summary.iloc[0]

    best_name = str(best_row["model"])
    best_stage = str(best_row["stage"])

    if best_stage == "pycaret_tuned":
        best_pipeline = tuned_pipelines[best_name]
    else:
        best_pipeline = model_by_name[best_name]

    print("\n=== MELHOR RESULTADO PYCARET ===")
    print(f"Modelo: {best_name}")
    print(f"Estágio: {best_stage}")
    print(f"F1 médio: {best_row['f1_mean']:.4f}")
    print(f"EER médio: {best_row['eer_mean']:.4f}")
    print(f"ROC-AUC médio: {best_row['roc_auc_mean']:.4f}")
    print(
        "Balanced Accuracy média: "
        f"{best_row['balanced_accuracy_mean']:.4f}"
    )
    print(
        "Latência p90 média: "
        f"{best_row['latency_p90_s_mean']:.3f} s"
    )

    # Treinar o pipeline vencedor em TODAS as janelas desta configuração.
    X_full = df[features]

    y_full = (
        df[TARGET_COLUMN]
        .astype(str)
        .eq(POSITIVE_LABEL)
        .astype(int)
        .to_numpy()
    )

    final_pipeline = clone(best_pipeline)

    final_pipeline.fit(
        X_full,
        y_full,
    )

    joblib.dump(
        final_pipeline,
        MODEL_PATH,
    )

    # Predições externas de todos os candidatos
    all_predictions = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    all_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    features_payload = {
        "feature_set": FEATURE_SET_NAME,
        "features": features,
        "target_original": TARGET_COLUMN,
        "target_binary": {
            "mouse": 0,
            "trackpad": 1,
        },
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP,
        "best_model": best_name,
        "best_stage": best_stage,
        "metrics": {
            "f1_mean": float(best_row["f1_mean"]),
            "eer_mean": float(best_row["eer_mean"]),
            "roc_auc_mean": float(best_row["roc_auc_mean"]),
            "balanced_accuracy_mean": float(
                best_row["balanced_accuracy_mean"]
            ),
        },
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
        "fase": "PyCaret - refinamento do detector mouse x trackpad",
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pycaret_version": pycaret.__version__,
        "dataset": DATASET_PATH.name,
        "target_user": TARGET_USER,
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP,
        "feature_set": FEATURE_SET_NAME,
        "n_features": len(features),
        "n_windows": len(df),
        "n_sessions": int(
            df["session_id"].nunique()
        ),
        "n_select": N_SELECT,
        "n_tune": N_TUNE,
        "n_tune_iter": N_TUNE_ITER,
        "ranking_internal_pycaret": "AUC",
        "ranking_final": (
            "menor EER; depois maior AUC; "
            "depois maior Balanced Accuracy; depois maior F1"
        ),
        "session_folds": fold_definitions,
        "best_model": {
            "name": best_name,
            "stage": best_stage,
            "f1_mean": float(best_row["f1_mean"]),
            "eer_mean": float(best_row["eer_mean"]),
            "roc_auc_mean": float(best_row["roc_auc_mean"]),
            "balanced_accuracy_mean": float(
                best_row["balanced_accuracy_mean"]
            ),
        },
        "methodological_notes": [
            "Wheel/scroll foi excluído das features principais para reduzir circularidade com o rótulo criado pelo front-end.",
            "A validação principal usa sessões inteiras; janelas da mesma session_id não são divididas entre treino e teste nos folds científicos.",
            "PyCaret é usado como mecanismo AutoML de comparação e tuning; as métricas finais são recalculadas externamente nos folds por sessão.",
            "O rótulo device_context deriva da heurística do front-end e não é ground truth físico independente.",
            "O experimento é intrausuária (Bianca) e não deve ser descrito como detector universal de hardware.",
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
            default=str,
        )

    print("\n" + "=" * 88)
    print("PYCARET CONCLUÍDO")
    print("=" * 88)

    print(f"\nLeaderboard:\n{LEADERBOARD_PATH}")
    print(f"\nTop 5 avaliados:\n{OOF_BASE_PATH}")
    print(f"\nModelos tunados:\n{OOF_TUNED_PATH}")
    print(f"\nPredições OOF:\n{PREDICTIONS_PATH}")
    print(f"\nParâmetros de tuning:\n{TUNING_PATH}")
    print(f"\nModelo vencedor:\n{MODEL_PATH}")
    print(f"\nFeatures:\n{FEATURES_PATH}")
    print(f"\nManifesto:\n{MANIFEST_PATH}")

    print(
        "\n✅ Compare agora este resultado com o baseline:"
        "\n   XGBoost 25 eventos / 75% overlap"
        "\n   F1 = 0.7896"
        "\n   EER = 0.2836"
        "\n   AUC = 0.7760"
    )


if __name__ == "__main__":
    main()