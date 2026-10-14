"""
12B_finalizar_resultados_lstm_tensorflow.py

CYBELL-J — FASE 12B
FINALIZAÇÃO DOS RESULTADOS LSTM TENSORFLOW
SEM RETREINAR NENHUMA REDE

Motivo
------
O treinamento das 18 redes LSTM (6 perfis x 3 folds) já foi concluído.
O erro ocorreu apenas na etapa de agregação temporal porque, em alguns
folds/horizontes, a exigência de esperar o horizonte completo eliminou
todas as decisões de uma das classes (genuíno ou impostor).

Este script:
- reutiliza predicoes_oof.csv já salvo;
- NÃO importa TensorFlow;
- NÃO treina modelos;
- NÃO usa GPU;
- NÃO recria sequências;
- calcula 0/5/10/15/20 s;
- pula combinações sem as duas classes;
- exige 3 folds válidos para um horizonte poder ser eleito;
- mede cobertura das decisões;
- compara com o baseline clássico/PyCaret;
- gera o resultado global final da LSTM.

Entrada
-------
data/lstm_tensorflow_authentication/predicoes_oof.csv

Saídas
------
data/lstm_tensorflow_authentication/
    resultados_temporais_folds.csv
    resumo_temporal.csv
    melhor_horizonte_por_perfil.csv
    horizontes_invalidos.csv
    comparacao_lstm_vs_classico.csv
    resultado_global.csv
    manifesto_12b_finalizacao.json
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
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUT = (
    BASE_DIR
    / "data"
    / "lstm_tensorflow_authentication"
)

PREDICTIONS_PATH = (
    OUT
    / "predicoes_oof.csv"
)

TEMPORAL_FOLDS_PATH = (
    OUT
    / "resultados_temporais_folds.csv"
)

TEMPORAL_SUMMARY_PATH = (
    OUT
    / "resumo_temporal.csv"
)

BEST_TEMPORAL_PATH = (
    OUT
    / "melhor_horizonte_por_perfil.csv"
)

INVALID_PATH = (
    OUT
    / "horizontes_invalidos.csv"
)

COMPARISON_PATH = (
    OUT
    / "comparacao_lstm_vs_classico.csv"
)

GLOBAL_PATH = (
    OUT
    / "resultado_global.csv"
)

MANIFEST_PATH = (
    OUT
    / "manifesto_12b_finalizacao.json"
)

CLASSIC_BASELINE_PATH = (
    BASE_DIR
    / "data"
    / "mouse_authentication_temporal"
    / "melhor_horizonte_por_perfil.csv"
)

HORIZONS = [
    0,
    5,
    10,
    15,
    20,
]

EXPECTED_FOLDS = 3

REQUIRE_FULL_HORIZON = True
MIN_SCORES = 2


# ============================================================
# MÉTRICAS
# ============================================================

def compute_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
):
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
        "eer_threshold": float(
            thresholds[
                idx
            ]
        ),
        "far_at_eer": far,
        "frr_at_eer": frr,
    }


def auth_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    finite = np.isfinite(
        scores
    )

    y_true = y_true[
        finite
    ]

    scores = scores[
        finite
    ]

    classes = np.unique(
        y_true
    )

    if len(
        classes
    ) < 2:
        return None

    eer_info = compute_eer(
        y_true,
        scores,
    )

    pred = (
        scores
        >= eer_info[
            "eer_threshold"
        ]
    ).astype(
        np.int8
    )

    return {
        "n_decisions": int(
            len(
                y_true
            )
        ),
        "n_genuine": int(
            np.sum(
                y_true
                == 1
            )
        ),
        "n_impostor": int(
            np.sum(
                y_true
                == 0
            )
        ),
        **eer_info,
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "f1_genuine_eer": float(
            f1_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "balanced_accuracy_eer": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
        "accuracy_eer": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "precision_genuine_eer": float(
            precision_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_genuine_eer": float(
            recall_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
    }


# ============================================================
# AGREGAÇÃO TEMPORAL RETROSPECTIVA
# ============================================================

def aggregate_session(
    session: pd.DataFrame,
    horizon_s: int,
):
    g = session.sort_values(
        [
            "sequence_end_ms",
            "sequence_start_ms",
            "sequence_index",
        ]
    ).reset_index(
        drop=True
    )

    times = g[
        "sequence_end_ms"
    ].to_numpy(
        dtype=np.int64
    )

    scores = g[
        "score_genuine"
    ].to_numpy(
        dtype=np.float32
    )

    labels = g[
        "y_true_genuine"
    ].to_numpy(
        dtype=np.int8
    )

    if len(
        np.unique(
            labels
        )
    ) != 1:
        raise RuntimeError(
            "Uma session_id contém mais de um rótulo."
        )

    if horizon_s == 0:
        return pd.DataFrame(
            {
                "y_true_genuine": labels,
                "score_eval": scores,
                "n_scores": np.ones(
                    len(
                        g
                    ),
                    dtype=np.int16,
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
    out_scores = []
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

        if n < MIN_SCORES:
            continue

        if (
            REQUIRE_FULL_HORIZON
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

        out_scores.append(
            float(
                score
            )
        )

        out_n.append(
            int(
                n
            )
        )

    return pd.DataFrame(
        {
            "y_true_genuine": out_y,
            "score_eval": out_scores,
            "n_scores": out_n,
        }
    )


# ============================================================
# AVALIAÇÃO
# ============================================================

def evaluate_temporal(
    predictions: pd.DataFrame,
):
    valid_rows = []
    invalid_rows = []

    baseline_counts = (
        predictions.groupby(
            [
                "protected_user",
                "fold",
            ]
        )
        .size()
        .to_dict()
    )

    groups = predictions.groupby(
        [
            "protected_user",
            "fold",
        ],
        sort=True,
    )

    for (
        protected_user,
        fold,
    ), fold_df in groups:

        print(
            f"\n{protected_user} | fold {fold}"
        )

        baseline_n = int(
            baseline_counts[
                (
                    protected_user,
                    fold,
                )
            ]
        )

        for horizon_s in HORIZONS:
            parts = []

            for _, session in fold_df.groupby(
                "session_id",
                sort=False,
            ):
                part = aggregate_session(
                    session,
                    horizon_s=horizon_s,
                )

                if not part.empty:
                    parts.append(
                        part
                    )

            if not parts:
                invalid_rows.append(
                    {
                        "protected_user": protected_user,
                        "fold": int(
                            fold
                        ),
                        "horizon_s": int(
                            horizon_s
                        ),
                        "reason": "sem_decisoes_apos_filtro_temporal",
                        "n_decisions": 0,
                        "n_genuine": 0,
                        "n_impostor": 0,
                    }
                )

                print(
                    f"   {horizon_s:>2}s | INVÁLIDO: sem decisões"
                )

                continue

            decisions = pd.concat(
                parts,
                ignore_index=True,
            )

            n_genuine = int(
                (
                    decisions[
                        "y_true_genuine"
                    ]
                    == 1
                ).sum()
            )

            n_impostor = int(
                (
                    decisions[
                        "y_true_genuine"
                    ]
                    == 0
                ).sum()
            )

            m = auth_metrics(
                decisions[
                    "y_true_genuine"
                ].to_numpy(),
                decisions[
                    "score_eval"
                ].to_numpy(),
            )

            if m is None:
                invalid_rows.append(
                    {
                        "protected_user": protected_user,
                        "fold": int(
                            fold
                        ),
                        "horizon_s": int(
                            horizon_s
                        ),
                        "reason": "apenas_uma_classe_apos_filtro_temporal",
                        "n_decisions": int(
                            len(
                                decisions
                            )
                        ),
                        "n_genuine": n_genuine,
                        "n_impostor": n_impostor,
                    }
                )

                print(
                    f"   {horizon_s:>2}s | INVÁLIDO: "
                    f"genuíno={n_genuine}, impostor={n_impostor}"
                )

                continue

            valid_rows.append(
                {
                    "protected_user": str(
                        protected_user
                    ),
                    "fold": int(
                        fold
                    ),
                    "horizon_s": int(
                        horizon_s
                    ),
                    "coverage_ratio": float(
                        len(
                            decisions
                        )
                        / baseline_n
                    ),
                    "mean_scores_per_decision": float(
                        decisions[
                            "n_scores"
                        ].mean()
                    ),
                    **m,
                }
            )

            print(
                f"   {horizon_s:>2}s | "
                f"EER={m['eer']:.4f} | "
                f"AUC={m['roc_auc']:.4f} | "
                f"F1={m['f1_genuine_eer']:.4f} | "
                f"cobertura={len(decisions)/baseline_n:.3f}"
            )

    return (
        pd.DataFrame(
            valid_rows
        ),
        pd.DataFrame(
            invalid_rows
        ),
    )


# ============================================================
# RESUMO / SELEÇÃO
# ============================================================

def summarize_temporal(
    temporal_folds: pd.DataFrame,
):
    metrics = [
        "eer",
        "far_at_eer",
        "frr_at_eer",
        "roc_auc",
        "f1_genuine_eer",
        "balanced_accuracy_eer",
        "accuracy_eer",
        "precision_genuine_eer",
        "recall_genuine_eer",
        "coverage_ratio",
        "mean_scores_per_decision",
    ]

    rows = []

    for (
        protected_user,
        horizon_s,
    ), group in temporal_folds.groupby(
        [
            "protected_user",
            "horizon_s",
        ],
        sort=True,
    ):
        row = {
            "protected_user": protected_user,
            "horizon_s": int(
                horizon_s
            ),
            "n_valid_folds": int(
                group[
                    "fold"
                ].nunique()
            ),
        }

        for metric in metrics:
            v = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            )

            row[
                f"{metric}_mean"
            ] = float(
                v.mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                v.std(
                    ddof=0
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def choose_best(
    summary: pd.DataFrame,
):
    winners = []

    for user, group in summary.groupby(
        "protected_user",
        sort=True,
    ):
        eligible = group[
            group[
                "n_valid_folds"
            ]
            .eq(
                EXPECTED_FOLDS
            )
        ].copy()

        if eligible.empty:
            raise RuntimeError(
                f"{user}: nenhum horizonte possui "
                f"{EXPECTED_FOLDS} folds válidos."
            )

        ranked = eligible.sort_values(
            [
                "eer_mean",
                "roc_auc_mean",
                "f1_genuine_eer_mean",
                "coverage_ratio_mean",
                "horizon_s",
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
            ],
        )

        winners.append(
            ranked.iloc[
                0
            ].copy()
        )

    return pd.DataFrame(
        winners
    ).reset_index(
        drop=True
    )


# ============================================================
# COMPARAÇÃO COM CLÁSSICO
# ============================================================

def compare_with_classic(
    best: pd.DataFrame,
):
    if not CLASSIC_BASELINE_PATH.exists():
        print(
            "\nBaseline clássico não encontrado; comparação pulada."
        )
        return pd.DataFrame()

    classic = pd.read_csv(
        CLASSIC_BASELINE_PATH,
        low_memory=False,
    )

    if "protected_user" not in classic.columns:
        print(
            "\nBaseline clássico sem protected_user; comparação pulada."
        )
        return pd.DataFrame()

    classic_cols = [
        c
        for c in [
            "protected_user",
            "candidate",
            "strategy",
            "horizon_s",
            "eer_mean",
            "roc_auc_mean",
            "f1_genuine_eer_mean",
            "balanced_accuracy_eer_mean",
        ]
        if c in classic.columns
    ]

    classic_small = classic[
        classic_cols
    ].copy()

    classic_small = classic_small.rename(
        columns={
            c: (
                c
                if c == "protected_user"
                else f"classic_{c}"
            )
            for c in classic_small.columns
        }
    )

    lstm_small = best[
        [
            "protected_user",
            "horizon_s",
            "eer_mean",
            "roc_auc_mean",
            "f1_genuine_eer_mean",
            "balanced_accuracy_eer_mean",
            "coverage_ratio_mean",
        ]
    ].rename(
        columns={
            "horizon_s": "lstm_horizon_s",
            "eer_mean": "lstm_eer_mean",
            "roc_auc_mean": "lstm_roc_auc_mean",
            "f1_genuine_eer_mean": "lstm_f1_genuine_eer_mean",
            "balanced_accuracy_eer_mean": (
                "lstm_balanced_accuracy_eer_mean"
            ),
            "coverage_ratio_mean": "lstm_coverage_ratio_mean",
        }
    )

    comparison = lstm_small.merge(
        classic_small,
        on="protected_user",
        how="left",
    )

    if "classic_eer_mean" in comparison.columns:
        comparison[
            "eer_delta_lstm_minus_classic"
        ] = (
            comparison[
                "lstm_eer_mean"
            ]
            - comparison[
                "classic_eer_mean"
            ]
        )

        comparison[
            "winner_lower_eer"
        ] = np.where(
            comparison[
                "lstm_eer_mean"
            ]
            < comparison[
                "classic_eer_mean"
            ],
            "LSTM_TENSORFLOW",
            "CLASSICO_PYCARET",
        )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "="
        * 112
    )

    print(
        "CYBELL-J — FASE 12B — "
        "FINALIZANDO RESULTADOS LSTM TENSORFLOW"
    )

    print(
        "="
        * 112
    )

    print(
        "\nSem treinamento."
        "\nSem TensorFlow."
        "\nSem GPU."
        "\nUsando apenas as predições OOF já salvas."
    )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predições OOF não encontradas:\n"
            f"{PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        low_memory=False,
        dtype={
            "protected_user": "string",
            "fold": "int8",
            "sequence_index": "int64",
            "session_id": "string",
            "actual_user": "string",
            "sequence_start_ms": "int64",
            "sequence_end_ms": "int64",
            "y_true_genuine": "int8",
            "score_genuine": "float32",
        },
    )

    print(
        f"\nPredições carregadas: "
        f"{len(predictions):,}"
        .replace(
            ",",
            ".",
        )
    )

    print(
        f"Perfis: "
        f"{predictions['protected_user'].nunique()}"
    )

    print(
        f"Folds: "
        f"{sorted(predictions['fold'].unique().tolist())}"
    )

    print(
        "\n1) Recalculando agregação temporal com proteção "
        "contra folds de classe única..."
    )

    temporal_folds, invalid = evaluate_temporal(
        predictions
    )

    temporal_folds.to_csv(
        TEMPORAL_FOLDS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    invalid.to_csv(
        INVALID_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_temporal(
        temporal_folds
    )

    summary.to_csv(
        TEMPORAL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    best = choose_best(
        summary
    )

    best.to_csv(
        BEST_TEMPORAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "2) MELHOR HORIZONTE LSTM POR PERFIL "
        "(somente horizontes com 3 folds válidos)"
    )

    print(
        "="
        * 112
    )

    print(
        best[
            [
                "protected_user",
                "horizon_s",
                "n_valid_folds",
                "eer_mean",
                "roc_auc_mean",
                "f1_genuine_eer_mean",
                "balanced_accuracy_eer_mean",
                "coverage_ratio_mean",
            ]
        ].to_string(
            index=False
        )
    )

    comparison = compare_with_classic(
        best
    )

    if not comparison.empty:
        comparison.to_csv(
            COMPARISON_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n"
            + "="
            * 112
        )

        print(
            "3) LSTM TENSORFLOW x CLÁSSICO/PYCARET"
        )

        print(
            "="
            * 112
        )

        show = [
            c
            for c in [
                "protected_user",
                "lstm_horizon_s",
                "lstm_eer_mean",
                "classic_horizon_s",
                "classic_eer_mean",
                "eer_delta_lstm_minus_classic",
                "winner_lower_eer",
            ]
            if c in comparison.columns
        ]

        print(
            comparison[
                show
            ].to_string(
                index=False
            )
        )

    global_row = {
        "model": "LSTM_TENSORFLOW",
        "profiles": int(
            best[
                "protected_user"
            ].nunique()
        ),
        "eer_mean_profiles": float(
            best[
                "eer_mean"
            ].mean()
        ),
        "eer_median_profiles": float(
            best[
                "eer_mean"
            ].median()
        ),
        "auc_mean_profiles": float(
            best[
                "roc_auc_mean"
            ].mean()
        ),
        "f1_eer_mean_profiles": float(
            best[
                "f1_genuine_eer_mean"
            ].mean()
        ),
        "balanced_accuracy_eer_mean_profiles": float(
            best[
                "balanced_accuracy_eer_mean"
            ].mean()
        ),
        "coverage_mean_profiles": float(
            best[
                "coverage_ratio_mean"
            ].mean()
        ),
    }

    global_df = pd.DataFrame(
        [
            global_row
        ]
    )

    global_df.to_csv(
        GLOBAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "4) RESULTADO GLOBAL LSTM"
    )

    print(
        "="
        * 112
    )

    print(
        global_df.to_string(
            index=False
        )
    )

    print(
        "\nCombinações temporais inválidas "
        f"(esperado em sessões curtas): {len(invalid)}"
    )

    if not invalid.empty:
        print(
            invalid.to_string(
                index=False
            )
        )

    manifest = {
        "projeto": "CYBELL-J",
        "fase": (
            "12B - finalizacao dos resultados "
            "LSTM TensorFlow sem retreinamento"
        ),
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_predictions": str(
            PREDICTIONS_PATH
        ),
        "training_repeated": False,
        "tensorflow_required": False,
        "gpu_used": False,
        "horizons_seconds": HORIZONS,
        "require_full_horizon": REQUIRE_FULL_HORIZON,
        "minimum_scores": MIN_SCORES,
        "selection_rule": (
            "Somente horizontes com 3 folds válidos podem vencer; "
            "menor EER médio, depois maior AUC/F1/cobertura."
        ),
        "global_result": global_row,
        "invalid_combinations": int(
            len(
                invalid
            )
        ),
        "notes": [
            "O erro original ocorreu porque algumas combinações horizonte/fold ficaram com apenas uma classe após exigir o horizonte completo.",
            "Essas combinações são registradas como inválidas e não entram no cálculo do EER.",
            "O horizonte vencedor por perfil deve possuir os três folds válidos, evitando selecionar um resultado artificialmente bom com cobertura incompleta.",
            "A comparação clássica usa os resultados anteriores do pipeline PyCaret/modelos clássicos.",
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

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "FASE 12B CONCLUÍDA"
    )

    print(
        "="
        * 112
    )

    print(
        f"\nResultados temporais:\n"
        f"{TEMPORAL_FOLDS_PATH}"
    )

    print(
        f"\nResumo temporal:\n"
        f"{TEMPORAL_SUMMARY_PATH}"
    )

    print(
        f"\nMelhor horizonte:\n"
        f"{BEST_TEMPORAL_PATH}"
    )

    print(
        f"\nHorizontes inválidos:\n"
        f"{INVALID_PATH}"
    )

    print(
        f"\nComparação LSTM x clássico/PyCaret:\n"
        f"{COMPARISON_PATH}"
    )

    print(
        f"\nResultado global:\n"
        f"{GLOBAL_PATH}"
    )


if __name__ == "__main__":
    main()