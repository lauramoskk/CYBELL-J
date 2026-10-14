"""
11C_agregacao_temporal_perfil_vs_subperfil.py

CYBELL-J — Experimento 11C
AGREGAÇÃO TEMPORAL:
PERFIL ÚNICO CONTROLADO x SUBPERFIL MOUSE

Objetivo
--------
Usar SOMENTE as predições OOF já produzidas no Experimento 11B para
testar se a agregação temporal de 5, 10, 15 ou 20 segundos melhora a
autenticação da Bianca e, especialmente, se estabiliza o Fold 3 / S06.

IMPORTANTE
----------
- NÃO retreina modelos.
- NÃO executa PyCaret novamente.
- NÃO usa GPU.
- Usa as predições dos modelos que FORAM SELECIONADOS PELO PYCARET no 11B.
- Processa um fold/repetição/estratégia por vez.
- Lê o CSV de predições em chunks e o particiona temporariamente no disco.
- Mantém uso de RAM baixo.
- A agregação é retrospectiva: nunca usa uma janela futura.
- A média temporal ocorre somente dentro da mesma session_id.

Entradas
--------
data/profile_vs_mouse_subprofile_controlled/predicoes_oof.csv
data/mouse_dataset/dataset_device_high_confidence.csv

Saídas
------
data/profile_vs_mouse_subprofile_temporal/
    resultados_fold_repeat_horizonte.csv
    resumo_temporal_estrategias.csv
    melhor_horizonte_por_estrategia.csv
    comparacao_final_temporal.csv
    comparacao_pareada_horizontes.csv
    detalhe_sessoes_bianca.csv
    manifesto_11c.json

Interpretação
-------------
O baseline "0 s" é a janela individual.
Os horizontes 5/10/15/20 s usam a média retrospectiva dos scores
genuínos dentro da mesma sessão.

O EER é recalculado em cada fold/repetição/horizonte.

Como as cinco repetições de cada fold compartilham o mesmo conjunto de
teste, elas medem estabilidade do treinamento/amostragem e NÃO devem ser
interpretadas como cinco testes independentes.
"""

from __future__ import annotations

import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    BASE_DIR
    / "data"
    / "profile_vs_mouse_subprofile_controlled"
)

PREDICTIONS_PATH = (
    SOURCE_DIR
    / "predicoes_oof.csv"
)

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "profile_vs_mouse_subprofile_temporal"
)

TEMP_DIR = (
    OUTPUT_DIR
    / "_tmp_partitions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "resultados_fold_repeat_horizonte.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "resumo_temporal_estrategias.csv"
)

BEST_PATH = (
    OUTPUT_DIR
    / "melhor_horizonte_por_estrategia.csv"
)

FINAL_PATH = (
    OUTPUT_DIR
    / "comparacao_final_temporal.csv"
)

PAIRED_PATH = (
    OUTPUT_DIR
    / "comparacao_pareada_horizontes.csv"
)

SESSION_DETAIL_PATH = (
    OUTPUT_DIR
    / "detalhe_sessoes_bianca.csv"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "manifesto_11c.json"
)

WINDOW_SIZE = 50
OVERLAP_PCT = 75

HORIZONS_SECONDS = [
    0,
    5,
    10,
    15,
    20,
]

STRATEGIES = [
    "PERFIL_UNICO_CONTROLADO",
    "SUBPERFIL_MOUSE",
]

# O 11B mostrou PYCARET_SELECTED como vencedor nas duas estratégias.
CANDIDATE_FAMILY = "PYCARET_SELECTED"

CSV_CHUNK_SIZE = 30_000

MIN_PREDICTIONS_TEMPORAL = 2
REQUIRE_FULL_HORIZON = True


# ============================================================
# UTILIDADES
# ============================================================

def reset_outputs():

    for path in [
        RESULTS_PATH,
        SESSION_DETAIL_PATH,
    ]:
        if path.exists():
            path.unlink()

    if TEMP_DIR.exists():
        shutil.rmtree(
            TEMP_DIR,
            ignore_errors=True,
        )

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def append_csv(
    df: pd.DataFrame,
    path: Path,
):

    if df.empty:
        return

    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# TIMESTAMPS MÍNIMOS
# ============================================================

def load_timestamp_lookup() -> pd.DataFrame:
    """
    Carrega somente as colunas mínimas necessárias do dataset original.
    """

    usecols = [
        "window_id",
        "session_id",
        "window_size",
        "requested_overlap_pct",
        "device_context",
        "window_start_ms",
        "window_end_ms",
    ]

    chunks = []

    for chunk in pd.read_csv(
        DATASET_PATH,
        usecols=usecols,
        chunksize=50_000,
        low_memory=False,
    ):

        part = chunk[
            (
                chunk[
                    "window_size"
                ]
                == WINDOW_SIZE
            )
            & (
                chunk[
                    "requested_overlap_pct"
                ]
                == OVERLAP_PCT
            )
            & (
                chunk[
                    "device_context"
                ]
                .astype(str)
                .eq(
                    "mouse"
                )
            )
        ][
            [
                "window_id",
                "session_id",
                "window_start_ms",
                "window_end_ms",
            ]
        ].copy()

        if not part.empty:
            chunks.append(
                part
            )

        del chunk
        del part
        gc.collect()

    if not chunks:
        raise RuntimeError(
            "Nenhum timestamp mouse 50/75 encontrado."
        )

    lookup = pd.concat(
        chunks,
        ignore_index=True,
    )

    del chunks
    gc.collect()

    lookup[
        "window_id"
    ] = lookup[
        "window_id"
    ].astype(
        "string"
    )

    lookup[
        "session_id"
    ] = lookup[
        "session_id"
    ].astype(
        "string"
    )

    lookup[
        "window_start_ms"
    ] = pd.to_numeric(
        lookup[
            "window_start_ms"
        ],
        errors="raise",
    ).astype(
        "int64"
    )

    lookup[
        "window_end_ms"
    ] = pd.to_numeric(
        lookup[
            "window_end_ms"
        ],
        errors="raise",
    ).astype(
        "int64"
    )

    if lookup[
        "window_id"
    ].duplicated().any():
        raise RuntimeError(
            "window_id duplicado no lookup 50/75."
        )

    return lookup


# ============================================================
# PARTICIONAMENTO DAS PREDIÇÕES — BAIXA RAM
# ============================================================

def partition_pycaret_predictions():
    """
    Lê predicoes_oof.csv em chunks e escreve pequenos arquivos por:
        estratégia / fold / repetição

    Mantém apenas as predições da família PYCARET_SELECTED.
    """

    usecols = [
        "strategy",
        "candidate_family",
        "candidate",
        "fold",
        "repeat",
        "session_id",
        "window_id",
        "actual_user",
        "y_true_genuine",
        "score_genuine",
    ]

    expected_keys = set()

    for chunk in pd.read_csv(
        PREDICTIONS_PATH,
        usecols=usecols,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ):

        chunk = chunk[
            chunk[
                "candidate_family"
            ]
            .astype(str)
            .eq(
                CANDIDATE_FAMILY
            )
            & chunk[
                "strategy"
            ]
            .astype(str)
            .isin(
                STRATEGIES
            )
        ].copy()

        if chunk.empty:
            del chunk
            gc.collect()
            continue

        chunk[
            "fold"
        ] = pd.to_numeric(
            chunk[
                "fold"
            ],
            errors="raise",
        ).astype(
            "int8"
        )

        chunk[
            "repeat"
        ] = pd.to_numeric(
            chunk[
                "repeat"
            ],
            errors="raise",
        ).astype(
            "int8"
        )

        chunk[
            "score_genuine"
        ] = pd.to_numeric(
            chunk[
                "score_genuine"
            ],
            errors="raise",
        ).astype(
            "float32"
        )

        chunk[
            "y_true_genuine"
        ] = pd.to_numeric(
            chunk[
                "y_true_genuine"
            ],
            errors="raise",
        ).astype(
            "int8"
        )

        for (
            strategy,
            fold,
            repeat,
        ), part in chunk.groupby(
            [
                "strategy",
                "fold",
                "repeat",
            ],
            sort=False,
        ):

            key = (
                str(
                    strategy
                ),
                int(
                    fold
                ),
                int(
                    repeat
                ),
            )

            expected_keys.add(
                key
            )

            path = (
                TEMP_DIR
                / (
                    f"{strategy}"
                    f"__fold{int(fold)}"
                    f"__rep{int(repeat)}.csv"
                )
            )

            part.to_csv(
                path,
                mode="a",
                header=not path.exists(),
                index=False,
                encoding="utf-8-sig",
            )

        del chunk
        gc.collect()

    if not expected_keys:
        raise RuntimeError(
            "Nenhuma predição PYCARET_SELECTED encontrada no 11B."
        )

    return sorted(
        expected_keys
    )


# ============================================================
# MÉTRICAS
# ============================================================

def compute_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if len(
        np.unique(
            y_true
        )
    ) < 2:
        raise RuntimeError(
            "EER requer genuínos e impostores."
        )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=1,
    )

    fnr = (
        1.0
        - tpr
    )

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


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    valid = np.isfinite(
        scores
    )

    y_true = y_true[
        valid
    ]

    scores = scores[
        valid
    ]

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
    }


# ============================================================
# AGREGAÇÃO TEMPORAL
# ============================================================

def trailing_mean_session(
    session_df: pd.DataFrame,
    horizon_s: int,
) -> pd.DataFrame:
    """
    Retorna uma linha por decisão válida.
    Usa apenas scores anteriores e o score atual.
    """

    g = session_df.sort_values(
        [
            "window_end_ms",
            "window_start_ms",
            "window_id",
        ]
    ).reset_index(
        drop=True
    )

    times = g[
        "window_end_ms"
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

    # Baseline: sem agregação.
    if horizon_s == 0:

        out = g[
            [
                "session_id",
                "window_id",
                "actual_user",
                "y_true_genuine",
                "score_genuine",
            ]
        ].copy()

        out[
            "score_eval"
        ] = out[
            "score_genuine"
        ].astype(
            "float32"
        )

        out[
            "n_scores_aggregated"
        ] = np.int16(
            1
        )

        return out[
            [
                "session_id",
                "window_id",
                "actual_user",
                "y_true_genuine",
                "score_eval",
                "n_scores_aggregated",
            ]
        ]

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

    rows = []

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

        n_scores = (
            i
            - left
            + 1
        )

        if (
            n_scores
            < MIN_PREDICTIONS_TEMPORAL
        ):
            continue

        if REQUIRE_FULL_HORIZON:

            if (
                now
                - first_time
                < horizon_ms
            ):
                continue

        sum_scores = (
            cumulative[
                i + 1
            ]
            - cumulative[
                left
            ]
        )

        rows.append(
            {
                "session_id": str(
                    g.loc[
                        i,
                        "session_id",
                    ]
                ),
                "window_id": str(
                    g.loc[
                        i,
                        "window_id",
                    ]
                ),
                "actual_user": str(
                    g.loc[
                        i,
                        "actual_user",
                    ]
                ),
                "y_true_genuine": int(
                    labels[
                        i
                    ]
                ),
                "score_eval": float(
                    sum_scores
                    / n_scores
                ),
                "n_scores_aggregated": int(
                    n_scores
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def aggregate_group(
    group_df: pd.DataFrame,
    horizon_s: int,
) -> pd.DataFrame:

    parts = []

    for _, session in group_df.groupby(
        "session_id",
        sort=False,
    ):

        part = trailing_mean_session(
            session,
            horizon_s=horizon_s,
        )

        if not part.empty:
            parts.append(
                part
            )

    if not parts:
        return pd.DataFrame()

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts
    gc.collect()

    return result


# ============================================================
# DETALHE DA SESSÃO GENUÍNA DA BIANCA
# ============================================================

def build_genuine_session_detail(
    decisions: pd.DataFrame,
    strategy: str,
    candidate: str,
    fold: int,
    repeat: int,
    horizon_s: int,
    threshold: float,
) -> pd.DataFrame:

    genuine = decisions[
        decisions[
            "y_true_genuine"
        ]
        .eq(
            1
        )
    ].copy()

    if genuine.empty:
        return pd.DataFrame()

    rows = []

    for session_id, session in genuine.groupby(
        "session_id",
        sort=True,
    ):

        scores = session[
            "score_eval"
        ].to_numpy(
            dtype=np.float32
        )

        accepted = (
            scores
            >= threshold
        )

        rows.append(
            {
                "strategy": strategy,
                "candidate": candidate,
                "fold": fold,
                "repeat": repeat,
                "horizon_s": horizon_s,
                "session_id": str(
                    session_id
                ),
                "n_decisions": int(
                    len(
                        session
                    )
                ),
                "mean_genuine_score": float(
                    np.mean(
                        scores
                    )
                ),
                "median_genuine_score": float(
                    np.median(
                        scores
                    )
                ),
                "acceptance_rate_at_fold_eer_threshold": float(
                    np.mean(
                        accepted
                    )
                ),
                "rejection_rate_at_fold_eer_threshold": float(
                    1.0
                    - np.mean(
                        accepted
                    )
                ),
                "mean_scores_aggregated": float(
                    session[
                        "n_scores_aggregated"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RESUMOS
# ============================================================

def summarize(
    results: pd.DataFrame,
) -> pd.DataFrame:

    metric_cols = [
        "eer",
        "far_at_eer",
        "frr_at_eer",
        "roc_auc",
        "f1_genuine_eer",
        "balanced_accuracy_eer",
        "accuracy_eer",
        "n_decisions",
        "mean_scores_per_decision",
    ]

    rows = []

    for (
        strategy,
        horizon_s,
    ), group in results.groupby(
        [
            "strategy",
            "horizon_s",
        ],
        sort=True,
    ):

        row = {
            "strategy": strategy,
            "candidate_family": CANDIDATE_FAMILY,
            "horizon_s": int(
                horizon_s
            ),
            "n_fold_repeat_results": int(
                len(
                    group
                )
            ),
        }

        for metric in metric_cols:

            values = pd.to_numeric(
                group[
                    metric
                ],
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
                values.std(
                    ddof=0
                )
            )

            row[
                f"{metric}_median"
            ] = float(
                values.median()
            )

        # Robustez: primeiro média as 5 repetições dentro de cada fold.
        fold_means = (
            group.groupby(
                "fold"
            )[
                "eer"
            ]
            .mean()
        )

        row[
            "eer_worst_fold_mean"
        ] = float(
            fold_means.max()
        )

        row[
            "eer_best_fold_mean"
        ] = float(
            fold_means.min()
        )

        row[
            "eer_fold_range"
        ] = float(
            fold_means.max()
            - fold_means.min()
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "strategy",
            "eer_mean",
            "roc_auc_mean",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


def choose_best_per_strategy(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    winners = []

    for strategy, group in summary.groupby(
        "strategy",
        sort=True,
    ):

        ranked = group.sort_values(
            [
                "eer_mean",
                "eer_worst_fold_mean",
                "roc_auc_mean",
                "balanced_accuracy_eer_mean",
                "horizon_s",
            ],
            ascending=[
                True,
                True,
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


def paired_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for horizon_s, group in results.groupby(
        "horizon_s",
        sort=True,
    ):

        wide = group.pivot_table(
            index=[
                "fold",
                "repeat",
            ],
            columns="strategy",
            values="eer",
            aggfunc="first",
        )

        if not all(
            strategy
            in wide.columns
            for strategy in STRATEGIES
        ):
            continue

        delta = (
            wide[
                "PERFIL_UNICO_CONTROLADO"
            ]
            - wide[
                "SUBPERFIL_MOUSE"
            ]
        ).dropna()

        rows.append(
            {
                "horizon_s": int(
                    horizon_s
                ),
                "n_pairs": int(
                    len(
                        delta
                    )
                ),
                "eer_delta_unique_minus_subprofile_mean": float(
                    delta.mean()
                ),
                "eer_delta_unique_minus_subprofile_median": float(
                    delta.median()
                ),
                "unique_wins_lower_eer": int(
                    (
                        delta
                        < 0
                    ).sum()
                ),
                "subprofile_wins_lower_eer": int(
                    (
                        delta
                        > 0
                    ).sum()
                ),
                "ties": int(
                    (
                        delta
                        == 0
                    ).sum()
                ),
                "interpretation": (
                    "delta < 0 favorece PERFIL_UNICO_CONTROLADO; "
                    "delta > 0 favorece SUBPERFIL_MOUSE"
                ),
            }
        )

    return pd.DataFrame(
        rows
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
        "CYBELL-J — EXPERIMENTO 11C — "
        "AGREGACAO TEMPORAL PERFIL UNICO x SUBPERFIL MOUSE"
    )

    print(
        "="
        * 112
    )

    print(
        "\nPerfil de recursos:"
        "\n- sem treinamento"
        "\n- sem PyCaret nesta etapa"
        "\n- usa predições dos modelos selecionados pelo PyCaret no 11B"
        "\n- sem GPU"
        "\n- CSV lido em chunks"
        "\n- processamento por fold/repetição"
        "\n- baixa RAM"
    )

    for required in [
        PREDICTIONS_PATH,
        DATASET_PATH,
    ]:

        if not required.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado:\n"
                f"{required}"
            )

    reset_outputs()

    print(
        "\n1) Carregando lookup mínimo de timestamps..."
    )

    timestamp_lookup = (
        load_timestamp_lookup()
    )

    print(
        f"   timestamps: "
        f"{len(timestamp_lookup):,}"
        .replace(
            ",",
            ".",
        )
    )

    print(
        "\n2) Particionando somente as predições "
        "PYCARET_SELECTED do 11B..."
    )

    keys = partition_pycaret_predictions()

    print(
        f"   grupos encontrados: "
        f"{len(keys)}"
    )

    expected = (
        len(
            STRATEGIES
        )
        * 3
        * 5
    )

    if len(
        keys
    ) != expected:

        print(
            f"   AVISO: esperávamos {expected} grupos, "
            f"mas encontramos {len(keys)}."
        )

    result_rows = []

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "3) AVALIANDO 0 / 5 / 10 / 15 / 20 SEGUNDOS"
    )

    print(
        "="
        * 112
    )

    for index, (
        strategy,
        fold,
        repeat,
    ) in enumerate(
        keys,
        start=1,
    ):

        path = (
            TEMP_DIR
            / (
                f"{strategy}"
                f"__fold{fold}"
                f"__rep{repeat}.csv"
            )
        )

        print(
            f"\n[{index}/{len(keys)}] "
            f"{strategy} | fold {fold} | repetição {repeat}"
        )

        group = pd.read_csv(
            path,
            low_memory=False,
            dtype={
                "strategy": "string",
                "candidate_family": "string",
                "candidate": "string",
                "fold": "int8",
                "repeat": "int8",
                "session_id": "string",
                "window_id": "string",
                "actual_user": "string",
                "y_true_genuine": "int8",
                "score_genuine": "float32",
            },
        )

        candidate_values = (
            group[
                "candidate"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        candidate = (
            candidate_values[
                0
            ]
            if candidate_values
            else "PyCaret_selected"
        )

        group = group.merge(
            timestamp_lookup,
            on=[
                "window_id",
                "session_id",
            ],
            how="left",
            validate="many_to_one",
        )

        if group[
            "window_end_ms"
        ].isna().any():

            raise RuntimeError(
                f"Timestamp ausente em "
                f"{strategy}/fold{fold}/rep{repeat}."
            )

        group[
            "window_start_ms"
        ] = group[
            "window_start_ms"
        ].astype(
            "int64"
        )

        group[
            "window_end_ms"
        ] = group[
            "window_end_ms"
        ].astype(
            "int64"
        )

        for horizon_s in HORIZONS_SECONDS:

            decisions = aggregate_group(
                group,
                horizon_s=horizon_s,
            )

            if decisions.empty:
                print(
                    f"   {horizon_s:>2}s: sem decisões válidas"
                )
                continue

            m = calculate_metrics(
                decisions[
                    "y_true_genuine"
                ].to_numpy(),
                decisions[
                    "score_eval"
                ].to_numpy(),
            )

            row = {
                "strategy": strategy,
                "candidate_family": CANDIDATE_FAMILY,
                "candidate": candidate,
                "fold": int(
                    fold
                ),
                "repeat": int(
                    repeat
                ),
                "horizon_s": int(
                    horizon_s
                ),
                "mean_scores_per_decision": float(
                    decisions[
                        "n_scores_aggregated"
                    ].mean()
                ),
                **m,
            }

            result_rows.append(
                row
            )

            append_csv(
                pd.DataFrame(
                    [
                        row
                    ]
                ),
                RESULTS_PATH,
            )

            detail = build_genuine_session_detail(
                decisions=decisions,
                strategy=strategy,
                candidate=candidate,
                fold=fold,
                repeat=repeat,
                horizon_s=horizon_s,
                threshold=m[
                    "eer_threshold"
                ],
            )

            append_csv(
                detail,
                SESSION_DETAIL_PATH,
            )

            print(
                f"   {horizon_s:>2}s | "
                f"EER={m['eer']:.4f} | "
                f"AUC={m['roc_auc']:.4f} | "
                f"F1={m['f1_genuine_eer']:.4f}"
            )

            del decisions
            del detail

            gc.collect()

        del group
        gc.collect()

    results = pd.DataFrame(
        result_rows
    )

    if results.empty:
        raise RuntimeError(
            "Nenhum resultado temporal foi calculado."
        )

    summary = summarize(
        results
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    best = choose_best_per_strategy(
        summary
    )

    best.to_csv(
        BEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    final = best[
        [
            "strategy",
            "candidate_family",
            "horizon_s",
            "eer_mean",
            "eer_std",
            "eer_median",
            "eer_worst_fold_mean",
            "eer_fold_range",
            "roc_auc_mean",
            "f1_genuine_eer_mean",
            "balanced_accuracy_eer_mean",
        ]
    ].copy()

    final = final.sort_values(
        [
            "eer_mean",
            "eer_worst_fold_mean",
            "roc_auc_mean",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    final.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                final
            )
            + 1,
        ),
    )

    final.to_csv(
        FINAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    paired = paired_comparison(
        results
    )

    paired.to_csv(
        PAIRED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # RESULTADOS IMPRESSOS
    # ========================================================

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "4) RESUMO TEMPORAL POR ESTRATEGIA"
    )

    print(
        "="
        * 112
    )

    print(
        summary[
            [
                "strategy",
                "horizon_s",
                "eer_mean",
                "eer_std",
                "eer_median",
                "eer_worst_fold_mean",
                "roc_auc_mean",
                "f1_genuine_eer_mean",
                "balanced_accuracy_eer_mean",
            ]
        ]
        .sort_values(
            [
                "strategy",
                "horizon_s",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "5) MELHOR HORIZONTE POR ESTRATEGIA"
    )

    print(
        "="
        * 112
    )

    print(
        best[
            [
                "strategy",
                "horizon_s",
                "eer_mean",
                "eer_std",
                "eer_median",
                "eer_worst_fold_mean",
                "roc_auc_mean",
                "f1_genuine_eer_mean",
                "balanced_accuracy_eer_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "6) COMPARACAO FINAL TEMPORAL"
    )

    print(
        "="
        * 112
    )

    print(
        final.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "7) COMPARACAO PAREADA POR HORIZONTE"
    )

    print(
        "="
        * 112
    )

    print(
        paired.to_string(
            index=False
        )
    )

    # ========================================================
    # FOCO FOLD 3 / BIANCA S06
    # ========================================================

    session_detail = pd.read_csv(
        SESSION_DETAIL_PATH,
        low_memory=False,
    )

    s06 = session_detail[
        session_detail[
            "session_id"
        ]
        .astype(str)
        .eq(
            "bianca_S06"
        )
    ].copy()

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "8) FOCO BIANCA_S06 — FOLD PROBLEMATICO"
    )

    print(
        "="
        * 112
    )

    if s06.empty:

        print(
            "Não encontrei bianca_S06 no detalhe."
        )

    else:

        s06_summary = (
            s06.groupby(
                [
                    "strategy",
                    "horizon_s",
                ],
                sort=True,
            )
            .agg(
                acceptance_rate_mean=(
                    "acceptance_rate_at_fold_eer_threshold",
                    "mean",
                ),
                rejection_rate_mean=(
                    "rejection_rate_at_fold_eer_threshold",
                    "mean",
                ),
                genuine_score_mean=(
                    "mean_genuine_score",
                    "mean",
                ),
            )
            .reset_index()
        )

        print(
            s06_summary.to_string(
                index=False
            )
        )

    winner = final.iloc[
        0
    ]

    print(
        "\nMELHOR ESTRATEGIA TEMPORAL:"
    )

    print(
        f"   {winner['strategy']}"
    )

    print(
        f"   Horizonte: "
        f"{int(winner['horizon_s'])} s"
    )

    print(
        f"   EER médio: "
        f"{winner['eer_mean']:.4f}"
    )

    print(
        f"   EER do pior fold (média das repetições): "
        f"{winner['eer_worst_fold_mean']:.4f}"
    )

    print(
        f"   AUC média: "
        f"{winner['roc_auc_mean']:.4f}"
    )

    # ========================================================
    # MANIFESTO
    # ========================================================

    manifest = {
        "projeto": "CYBELL-J",
        "fase": (
            "Experimento 11C - agregacao temporal "
            "perfil unico controlado versus subperfil mouse"
        ),
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_predictions": str(
            PREDICTIONS_PATH.name
        ),
        "source_experiment": "11B",
        "candidate_family": CANDIDATE_FAMILY,
        "pycaret_role": (
            "As predições analisadas são dos modelos selecionados "
            "pelo PyCaret no experimento 11B. "
            "Nenhum novo treinamento ocorre no 11C."
        ),
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP_PCT,
        "horizons_seconds": HORIZONS_SECONDS,
        "aggregation": {
            "method": "trailing_mean",
            "same_session_only": True,
            "uses_future_predictions": False,
            "require_full_horizon": REQUIRE_FULL_HORIZON,
            "minimum_predictions_temporal": (
                MIN_PREDICTIONS_TEMPORAL
            ),
        },
        "resource_profile": {
            "training": False,
            "pycaret_runtime": False,
            "gpu": False,
            "chunk_size_predictions": CSV_CHUNK_SIZE,
            "temporary_disk_partitions": True,
            "one_fold_repeat_at_a_time": True,
            "gc_between_horizons": True,
        },
        "best_temporal_strategy": {
            "strategy": str(
                winner[
                    "strategy"
                ]
            ),
            "horizon_s": int(
                winner[
                    "horizon_s"
                ]
            ),
            "eer_mean": float(
                winner[
                    "eer_mean"
                ]
            ),
            "eer_worst_fold_mean": float(
                winner[
                    "eer_worst_fold_mean"
                ]
            ),
            "roc_auc_mean": float(
                winner[
                    "roc_auc_mean"
                ]
            ),
        },
        "methodological_notes": [
            "As cinco repetições dentro do mesmo fold compartilham o mesmo teste e não são amostras independentes.",
            "A agregação temporal é aplicada separadamente dentro de cada session_id.",
            "Janelas sobrepostas produzem scores correlacionados; a média temporal deve ser interpretada como acumulação de evidência e não como combinação de observações independentes.",
            "O EER é recalculado dentro de cada fold/repetição/horizonte.",
            "A análise da sessão bianca_S06 é diagnóstica e busca medir se o horizonte temporal estabiliza o fold mais difícil.",
        ],
        "outputs": {
            "fold_repeat_horizon_results": RESULTS_PATH.name,
            "summary": SUMMARY_PATH.name,
            "best_per_strategy": BEST_PATH.name,
            "final_comparison": FINAL_PATH.name,
            "paired_by_horizon": PAIRED_PATH.name,
            "bianca_session_detail": SESSION_DETAIL_PATH.name,
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
        )

    # Limpa apenas partições temporárias.
    shutil.rmtree(
        TEMP_DIR,
        ignore_errors=True,
    )

    print(
        "\n"
        + "="
        * 112
    )

    print(
        "EXPERIMENTO 11C CONCLUIDO"
    )

    print(
        "="
        * 112
    )

    print(
        f"\nResultados por fold/repetição/horizonte:\n"
        f"{RESULTS_PATH}"
    )

    print(
        f"\nResumo:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nMelhor horizonte por estratégia:\n"
        f"{BEST_PATH}"
    )

    print(
        f"\nComparação final:\n"
        f"{FINAL_PATH}"
    )

    print(
        f"\nComparação pareada:\n"
        f"{PAIRED_PATH}"
    )

    print(
        f"\nDetalhe das sessões genuínas da Bianca:\n"
        f"{SESSION_DETAIL_PATH}"
    )

    print(
        f"\nManifesto:\n"
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()