"""
08_agregacao_temporal.py

CYBELL-J — Agregação temporal das previsões Mouse x Trackpad

Esta etapa NÃO retreina modelos e NÃO altera datasets.
Ela usa as predições já geradas pelos experimentos anteriores e pergunta:

    "Se, em vez de decidir por uma única janela, o sistema acumular
     evidência por 5, 10, 15 ou 20 segundos, o desempenho melhora?"

Modelos comparados:
1) Baseline XGBoost — 25 eventos / 75% overlap
2) Melhor modelo PyCaret salvo no experimento 07

Método principal:
- Dentro de cada fold e de cada sessão, ordena as janelas pelo tempo.
- Para cada instante de decisão, calcula a MÉDIA das probabilidades
  produzidas nos últimos T segundos.
- T = 5, 10, 15 e 20 s.
- Mantém folds separados; nunca mistura resultados de folds diferentes.
- A verdade primária continua sendo o device_context da janela atual.
- Também calcula um recorte "stable_only": só decisões cujo rótulo
  verdadeiro permaneceu constante durante todo o horizonte temporal.
  Isso ajuda a separar:
      a) erro em estado estável
      b) atraso/ruído em transições de dispositivo

Métricas:
- F1
- EER
- ROC-AUC
- Balanced Accuracy
- Accuracy
- Precision
- Recall

Saídas:
    data/mouse_temporal_aggregation/
        resultados_temporais_por_fold.csv
        resumo_agregacao_temporal.csv
        decisoes_temporais.csv
        manifesto_agregacao_temporal.json

IMPORTANTE:
O rótulo mouse/trackpad continua derivado da heurística do front-end.
Este experimento continua sendo uma prova de conceito intrausuária.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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

BASELINE_PRED_PATH = (
    BASE_DIR
    / "data"
    / "mouse_training_device"
    / "predicoes_dispositivo.csv"
)

PYCARET_PRED_PATH = (
    BASE_DIR
    / "data"
    / "mouse_training_device_pycaret"
    / "pycaret_predicoes_oof.csv"
)

PYCARET_INFO_PATH = (
    BASE_DIR
    / "data"
    / "mouse_training_device_pycaret"
    / "melhor_modelo_pycaret_features.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "mouse_temporal_aggregation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_RESULTS_PATH = OUTPUT_DIR / "resultados_temporais_por_fold.csv"
SUMMARY_PATH = OUTPUT_DIR / "resumo_agregacao_temporal.csv"
DECISIONS_PATH = OUTPUT_DIR / "decisoes_temporais.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifesto_agregacao_temporal.json"

TARGET_USER = "bianca"
WINDOW_SIZE = 25
OVERLAP = 75

HORIZONS_SECONDS = [5, 10, 15, 20]

POSITIVE_LABEL = "trackpad"

# Para considerar que realmente houve "agregação", exigimos pelo menos
# duas previsões dentro do horizonte.
MIN_PREDICTIONS_PER_DECISION = 2

# Somente decisões que já tiveram tempo para acumular um horizonte
# completo desde o início da sessão.
REQUIRE_FULL_HORIZON = True


# ============================================================
# MÉTRICAS
# ============================================================

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

    idx = int(
        np.nanargmin(
            np.abs(fpr - fnr)
        )
    )

    eer = float(
        (fpr[idx] + fnr[idx]) / 2.0
    )

    threshold = float(
        thresholds[idx]
    )

    return eer, threshold


def safe_auc(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
) -> float:

    if len(np.unique(y_true_binary)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true_binary,
            scores,
        )
    )


def calculate_metrics(
    data: pd.DataFrame,
    score_column: str,
) -> dict:

    if data.empty:
        return {
            "n_decisions": 0,
            "n_mouse": 0,
            "n_trackpad": 0,
            "f1": float("nan"),
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "roc_auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    y_true = data["y_true"].astype(int).to_numpy()

    scores = pd.to_numeric(
        data[score_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = np.isfinite(scores)

    y_true = y_true[valid]
    scores = scores[valid]

    if len(y_true) == 0:
        return {
            "n_decisions": 0,
            "n_mouse": 0,
            "n_trackpad": 0,
            "f1": float("nan"),
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "roc_auc": float("nan"),
            "balanced_accuracy": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    pred = (scores >= 0.5).astype(int)

    eer, eer_threshold = compute_eer(
        y_true,
        scores,
    )

    return {
        "n_decisions": int(len(y_true)),
        "n_mouse": int(
            np.sum(y_true == 0)
        ),
        "n_trackpad": int(
            np.sum(y_true == 1)
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "roc_auc": safe_auc(
            y_true,
            scores,
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
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
    }


# ============================================================
# CARREGAMENTO E NORMALIZAÇÃO
# ============================================================

def load_dataset_timestamps() -> pd.DataFrame:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado:\n{DATASET_PATH}"
        )

    usecols = [
        "user",
        "session_id",
        "window_id",
        "window_size",
        "requested_overlap_pct",
        "window_start_ms",
        "window_end_ms",
        "device_context",
    ]

    df = pd.read_csv(
        DATASET_PATH,
        usecols=usecols,
        low_memory=False,
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
            df["device_context"]
            .isin(["mouse", "trackpad"])
        )
    ].copy()

    duplicated = df["window_id"].duplicated(
        keep=False
    )

    if duplicated.any():
        examples = (
            df.loc[
                duplicated,
                "window_id",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "window_id não é único para a configuração "
            f"{WINDOW_SIZE}/{OVERLAP}. Exemplos: {examples}"
        )

    df["y_true_from_dataset"] = (
        df["device_context"]
        .eq(POSITIVE_LABEL)
        .astype(int)
    )

    return df


def load_baseline_predictions() -> pd.DataFrame:

    if not BASELINE_PRED_PATH.exists():
        raise FileNotFoundError(
            f"Predições do baseline não encontradas:\n{BASELINE_PRED_PATH}"
        )

    pred = pd.read_csv(
        BASELINE_PRED_PATH,
        low_memory=False,
    )

    required = {
        "model",
        "window_size",
        "overlap_pct",
        "fold",
        "session_id",
        "window_id",
        "proba_trackpad",
    }

    missing = sorted(
        required - set(pred.columns)
    )

    if missing:
        raise RuntimeError(
            f"Colunas ausentes no baseline: {missing}"
        )

    pred = pred[
        pred["model"]
        .astype(str)
        .eq("XGBoost")
        & (
            pred["window_size"]
            == WINDOW_SIZE
        )
        & (
            pred["overlap_pct"]
            == OVERLAP
        )
    ].copy()

    if pred.empty:
        raise RuntimeError(
            "Não encontrei predições XGBoost "
            f"{WINDOW_SIZE}/{OVERLAP} no baseline."
        )

    pred = pred.rename(
        columns={
            "proba_trackpad": "score_trackpad",
        }
    )

    pred["source_model"] = "baseline_XGBoost"

    return pred[
        [
            "source_model",
            "fold",
            "session_id",
            "window_id",
            "score_trackpad",
        ]
    ].copy()


def load_pycaret_best_predictions() -> tuple[pd.DataFrame, dict]:

    if not PYCARET_PRED_PATH.exists():
        raise FileNotFoundError(
            f"Predições PyCaret não encontradas:\n{PYCARET_PRED_PATH}"
        )

    if not PYCARET_INFO_PATH.exists():
        raise FileNotFoundError(
            f"Informações do melhor PyCaret não encontradas:\n"
            f"{PYCARET_INFO_PATH}"
        )

    with PYCARET_INFO_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        info = json.load(f)

    best_model = str(
        info["best_model"]
    )

    best_stage = str(
        info["best_stage"]
    )

    pred = pd.read_csv(
        PYCARET_PRED_PATH,
        low_memory=False,
    )

    required = {
        "stage",
        "model",
        "fold",
        "session_id",
        "window_id",
        "score_trackpad",
    }

    missing = sorted(
        required - set(pred.columns)
    )

    if missing:
        raise RuntimeError(
            f"Colunas ausentes nas predições PyCaret: {missing}"
        )

    pred = pred[
        pred["model"]
        .astype(str)
        .eq(best_model)
        & (
            pred["stage"]
            .astype(str)
            .eq(best_stage)
        )
    ].copy()

    if pred.empty:
        raise RuntimeError(
            "Não encontrei as predições do melhor modelo PyCaret:\n"
            f"modelo={best_model} | stage={best_stage}"
        )

    pred["source_model"] = (
        f"pycaret_{best_model}_{best_stage}"
    )

    return (
        pred[
            [
                "source_model",
                "fold",
                "session_id",
                "window_id",
                "score_trackpad",
            ]
        ].copy(),
        info,
    )


def attach_timestamps_and_truth(
    predictions: pd.DataFrame,
    dataset_meta: pd.DataFrame,
) -> pd.DataFrame:

    merged = predictions.merge(
        dataset_meta[
            [
                "window_id",
                "session_id",
                "window_start_ms",
                "window_end_ms",
                "y_true_from_dataset",
            ]
        ],
        on=[
            "window_id",
            "session_id",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_time = merged[
        "window_end_ms"
    ].isna()

    if missing_time.any():
        examples = (
            merged.loc[
                missing_time,
                "window_id",
            ]
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            "Algumas predições não puderam ser ligadas "
            f"ao dataset por window_id. Exemplos: {examples}"
        )

    merged["y_true"] = (
        merged["y_true_from_dataset"]
        .astype(int)
    )

    # Verificação simples do score
    score = pd.to_numeric(
        merged["score_trackpad"],
        errors="coerce",
    )

    if score.isna().any():
        raise RuntimeError(
            "Existem scores não numéricos nas predições."
        )

    score_min = float(score.min())
    score_max = float(score.max())

    if score_min < -1e-6 or score_max > 1 + 1e-6:
        print(
            "   ⚠️ Atenção: score fora de [0,1]. "
            f"min={score_min:.4f}, max={score_max:.4f}. "
            "AUC/EER continuam possíveis, mas threshold 0.5 "
            "pode não representar probabilidade."
        )

    merged["score_trackpad"] = score.astype(float)

    return merged


# ============================================================
# AGREGAÇÃO TEMPORAL
# ============================================================

def aggregate_one_group(
    group: pd.DataFrame,
    horizon_seconds: int,
) -> pd.DataFrame:
    """
    Média móvel temporal das probabilidades.

    Para cada janela atual i:
      - pega previsões com window_end_ms nos últimos T segundos;
      - calcula score médio;
      - mantém y_true da janela atual como verdade primária;
      - calcula se todo o horizonte permaneceu com o mesmo y_true.
    """

    g = group.sort_values(
        [
            "window_end_ms",
            "window_start_ms",
            "window_id",
        ]
    ).reset_index(drop=True)

    times = (
        pd.to_numeric(
            g["window_end_ms"],
            errors="raise",
        )
        .to_numpy(dtype=np.float64)
    )

    scores = (
        pd.to_numeric(
            g["score_trackpad"],
            errors="raise",
        )
        .to_numpy(dtype=np.float64)
    )

    labels = (
        g["y_true"]
        .astype(int)
        .to_numpy()
    )

    horizon_ms = float(
        horizon_seconds * 1000
    )

    score_cumsum = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(scores),
        ]
    )

    label_cumsum = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.cumsum(labels, dtype=np.int64),
        ]
    )

    left = 0
    rows = []

    session_first_time = float(
        times[0]
    )

    for i in range(len(g)):

        current_time = float(
            times[i]
        )

        while (
            left < i
            and current_time - times[left] > horizon_ms
        ):
            left += 1

        n_predictions = (
            i - left + 1
        )

        if (
            n_predictions
            < MIN_PREDICTIONS_PER_DECISION
        ):
            continue

        if REQUIRE_FULL_HORIZON:
            if (
                current_time
                - session_first_time
                < horizon_ms
            ):
                continue

        sum_scores = (
            score_cumsum[i + 1]
            - score_cumsum[left]
        )

        mean_score = float(
            sum_scores
            / n_predictions
        )

        sum_labels = int(
            label_cumsum[i + 1]
            - label_cumsum[left]
        )

        trackpad_fraction = float(
            sum_labels
            / n_predictions
        )

        stable = (
            sum_labels == 0
            or sum_labels == n_predictions
        )

        current_label = int(
            labels[i]
        )

        rows.append(
            {
                "source_model": g.loc[
                    i,
                    "source_model",
                ],
                "fold": int(
                    g.loc[i, "fold"]
                ),
                "session_id": str(
                    g.loc[i, "session_id"]
                ),
                "window_id": str(
                    g.loc[i, "window_id"]
                ),
                "window_end_ms": current_time,
                "horizon_s": int(
                    horizon_seconds
                ),
                "n_predictions_aggregated": int(
                    n_predictions
                ),
                "score_single": float(
                    scores[i]
                ),
                "score_aggregated": mean_score,
                "y_true": current_label,
                "truth_trackpad_fraction_horizon": trackpad_fraction,
                "stable_true_context": bool(
                    stable
                ),
            }
        )

    return pd.DataFrame(rows)


def aggregate_all(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    parts = []

    group_columns = [
        "source_model",
        "fold",
        "session_id",
    ]

    grouped = predictions.groupby(
        group_columns,
        sort=True,
    )

    for horizon in HORIZONS_SECONDS:

        print(
            f"   Agregando horizonte de {horizon}s..."
        )

        for _, group in grouped:
            result = aggregate_one_group(
                group,
                horizon_seconds=horizon,
            )

            if not result.empty:
                parts.append(result)

    if not parts:
        raise RuntimeError(
            "Nenhuma decisão temporal foi gerada."
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ============================================================
# AVALIAÇÃO POR FOLD
# ============================================================

def evaluate_single_window_by_fold(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        source_model,
        fold,
    ), group in predictions.groupby(
        [
            "source_model",
            "fold",
        ],
        sort=True,
    ):

        metrics = calculate_metrics(
            group,
            score_column="score_trackpad",
        )

        rows.append(
            {
                "source_model": source_model,
                "strategy": "single_window",
                "horizon_s": 0,
                "subset": "all",
                "fold": int(fold),
                **metrics,
                "mean_predictions_per_decision": 1.0,
                "stable_fraction_pct": float(
                    "nan"
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_temporal_by_fold(
    decisions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    group_cols = [
        "source_model",
        "horizon_s",
        "fold",
    ]

    for (
        source_model,
        horizon_s,
        fold,
    ), group in decisions.groupby(
        group_cols,
        sort=True,
    ):

        stable_fraction = float(
            group["stable_true_context"]
            .mean()
            * 100
        )

        mean_n = float(
            group["n_predictions_aggregated"]
            .mean()
        )

        # 1) Todas as decisões — inclui transições.
        metrics_all = calculate_metrics(
            group,
            score_column="score_aggregated",
        )

        rows.append(
            {
                "source_model": source_model,
                "strategy": "temporal_mean",
                "horizon_s": int(
                    horizon_s
                ),
                "subset": "all",
                "fold": int(fold),
                **metrics_all,
                "mean_predictions_per_decision": mean_n,
                "stable_fraction_pct": stable_fraction,
            }
        )

        # 2) Somente contextos estáveis no horizonte.
        stable_group = group[
            group["stable_true_context"]
        ].copy()

        metrics_stable = calculate_metrics(
            stable_group,
            score_column="score_aggregated",
        )

        rows.append(
            {
                "source_model": source_model,
                "strategy": "temporal_mean",
                "horizon_s": int(
                    horizon_s
                ),
                "subset": "stable_only",
                "fold": int(fold),
                **metrics_stable,
                "mean_predictions_per_decision": (
                    float(
                        stable_group[
                            "n_predictions_aggregated"
                        ].mean()
                    )
                    if not stable_group.empty
                    else float("nan")
                ),
                "stable_fraction_pct": stable_fraction,
            }
        )

    return pd.DataFrame(rows)


def summarize_fold_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "n_decisions",
        "n_mouse",
        "n_trackpad",
        "f1",
        "eer",
        "roc_auc",
        "balanced_accuracy",
        "accuracy",
        "precision",
        "recall",
        "mean_predictions_per_decision",
        "stable_fraction_pct",
    ]

    rows = []

    grouped = fold_results.groupby(
        [
            "source_model",
            "strategy",
            "horizon_s",
            "subset",
        ],
        dropna=False,
        sort=True,
    )

    for keys, group in grouped:

        (
            source_model,
            strategy,
            horizon_s,
            subset,
        ) = keys

        row = {
            "source_model": source_model,
            "strategy": strategy,
            "horizon_s": int(
                horizon_s
            ),
            "subset": subset,
            "n_folds": int(
                len(group)
            ),
        }

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                values.std(ddof=0)
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    # Melhores estratégias primeiro:
    # 1) menor EER
    # 2) maior AUC
    # 3) maior balanced accuracy
    summary = summary.sort_values(
        [
            "subset",
            "eer_mean",
            "roc_auc_mean",
            "balanced_accuracy_mean",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("CYBELL-J — AGREGAÇÃO TEMPORAL MOUSE x TRACKPAD")
    print("=" * 90)

    print(
        "\nEsta etapa NÃO retreina modelos."
        "\nEla reutiliza as predições OOF já produzidas."
    )

    print("\n1) Carregando timestamps e verdade do dataset...")

    dataset_meta = load_dataset_timestamps()

    print(
        f"   Janelas da configuração "
        f"{WINDOW_SIZE}/{OVERLAP}: "
        f"{len(dataset_meta):,}"
        .replace(",", ".")
    )

    print("\n2) Carregando baseline XGBoost...")

    baseline = load_baseline_predictions()

    print(
        f"   Predições: {len(baseline):,}"
        .replace(",", ".")
    )

    print("\n3) Carregando melhor PyCaret...")

    pycaret_pred, pycaret_info = (
        load_pycaret_best_predictions()
    )

    print(
        f"   Melhor PyCaret: "
        f"{pycaret_info['best_model']} | "
        f"{pycaret_info['best_stage']}"
    )

    print(
        f"   Predições: {len(pycaret_pred):,}"
        .replace(",", ".")
    )

    baseline = attach_timestamps_and_truth(
        baseline,
        dataset_meta,
    )

    pycaret_pred = attach_timestamps_and_truth(
        pycaret_pred,
        dataset_meta,
    )

    predictions = pd.concat(
        [
            baseline,
            pycaret_pred,
        ],
        ignore_index=True,
    )

    # Verifica classes/folds
    print("\n4) Modelos preparados:")

    for model_name, group in predictions.groupby(
        "source_model"
    ):
        print(
            f"   {model_name}: "
            f"{len(group):,} predições | "
            f"{group['fold'].nunique()} folds"
            .replace(",", ".")
        )

    print("\n5) Calculando baseline de janela única...")

    single_fold_results = (
        evaluate_single_window_by_fold(
            predictions
        )
    )

    print("\n6) Agregando temporalmente...")

    decisions = aggregate_all(
        predictions
    )

    decisions.to_csv(
        DECISIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n   Decisões temporais geradas: "
        f"{len(decisions):,}"
        .replace(",", ".")
    )

    print("\n7) Calculando métricas por fold...")

    temporal_fold_results = (
        evaluate_temporal_by_fold(
            decisions
        )
    )

    all_fold_results = pd.concat(
        [
            single_fold_results,
            temporal_fold_results,
        ],
        ignore_index=True,
    )

    all_fold_results.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_fold_results(
        all_fold_results
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # IMPRESSÃO
    # ========================================================

    print("\n" + "=" * 90)
    print("RESULTADO — TODAS AS DECISÕES")
    print("=" * 90)

    all_subset = summary[
        summary["subset"].eq("all")
    ].copy()

    display_cols = [
        "source_model",
        "strategy",
        "horizon_s",
        "f1_mean",
        "eer_mean",
        "roc_auc_mean",
        "balanced_accuracy_mean",
        "mean_predictions_per_decision_mean",
    ]

    print(
        all_subset[
            display_cols
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print("RESULTADO — SOMENTE CONTEXTO ESTÁVEL")
    print("=" * 90)

    stable_subset = summary[
        summary["subset"].eq(
            "stable_only"
        )
    ].copy()

    if not stable_subset.empty:
        stable_display_cols = [
            "source_model",
            "horizon_s",
            "f1_mean",
            "eer_mean",
            "roc_auc_mean",
            "balanced_accuracy_mean",
            "stable_fraction_pct_mean",
        ]

        print(
            stable_subset[
                stable_display_cols
            ].to_string(
                index=False
            )
        )

    # Melhor temporal em todas as decisões
    temporal_all = summary[
        (
            summary["strategy"]
            .eq("temporal_mean")
        )
        & (
            summary["subset"]
            .eq("all")
        )
    ].copy()

    if not temporal_all.empty:
        temporal_all = temporal_all.sort_values(
            [
                "eer_mean",
                "roc_auc_mean",
                "balanced_accuracy_mean",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )

        best = temporal_all.iloc[0]

        print("\n" + "=" * 90)
        print("MELHOR AGREGAÇÃO TEMPORAL — TODAS AS DECISÕES")
        print("=" * 90)

        print(
            f"\nModelo: {best['source_model']}"
        )
        print(
            f"Horizonte: {int(best['horizon_s'])} s"
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
            "Balanced Accuracy média: "
            f"{best['balanced_accuracy_mean']:.4f}"
        )
        print(
            "Previsões agregadas por decisão: "
            f"{best['mean_predictions_per_decision_mean']:.1f}"
        )

    # ========================================================
    # MANIFESTO
    # ========================================================

    manifest = {
        "projeto": "CYBELL-J",
        "fase": "Agregação temporal do detector mouse x trackpad",
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": DATASET_PATH.name,
        "target_user": TARGET_USER,
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP,
        "horizons_seconds": HORIZONS_SECONDS,
        "aggregation": "trailing mean of model scores",
        "truth_primary": (
            "device_context da janela atual"
        ),
        "stable_subset_definition": (
            "todas as janelas verdadeiras dentro do "
            "horizonte possuem a mesma classe"
        ),
        "require_full_horizon": REQUIRE_FULL_HORIZON,
        "min_predictions_per_decision": MIN_PREDICTIONS_PER_DECISION,
        "models": sorted(
            predictions[
                "source_model"
            ].unique().tolist()
        ),
        "outputs": {
            "fold_results": FOLD_RESULTS_PATH.name,
            "summary": SUMMARY_PATH.name,
            "decisions": DECISIONS_PATH.name,
        },
        "methodological_notes": [
            "Nenhum modelo é retreinado nesta etapa.",
            "A agregação é feita separadamente dentro de fold e session_id.",
            "A decisão temporal usa apenas previsões passadas e a previsão atual; não usa previsões futuras.",
            "O resultado 'all' inclui períodos de transição de rótulo e portanto incorpora atraso de adaptação.",
            "O resultado 'stable_only' mede desempenho quando o contexto verdadeiro permaneceu constante em todo o horizonte.",
            "O rótulo device_context deriva da heurística do front-end e não constitui ground truth físico independente.",
            "Este experimento permanece intrausuária e não valida generalização universal de hardware.",
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

    print("\n" + "=" * 90)
    print("AGREGAÇÃO TEMPORAL CONCLUÍDA")
    print("=" * 90)

    print(
        f"\nResultados por fold:\n"
        f"{FOLD_RESULTS_PATH}"
    )

    print(
        f"\nResumo:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nDecisões temporais:\n"
        f"{DECISIONS_PATH}"
    )

    print(
        f"\nManifesto:\n"
        f"{MANIFEST_PATH}"
    )

    print(
        "\n✅ Agora compare os horizontes 5/10/15/20 s "
        "com a janela única."
    )


if __name__ == "__main__":
    main()