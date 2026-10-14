"""
10C_agregacao_temporal_autenticacao.py

CYBELL-J — Experimento 10C
AGREGAÇÃO TEMPORAL DA AUTENTICAÇÃO CONTÍNUA 1:1

Objetivo
--------
Usar SOMENTE as predições OOF já geradas no Experimento 10B e responder:

    "Se o sistema acumular evidência comportamental durante
     5, 10, 15 ou 20 segundos, o EER de autenticação melhora?"

IMPORTANTE
----------
- Este script NÃO retreina modelos.
- Este script NÃO usa PyCaret.
- Este script NÃO usa GPU.
- Ele foi desenhado para consumir POUCA RAM.
- Lê o arquivo grande de predições em CHUNKS.
- Processa UMA usuária protegida por vez.
- Carrega do dataset somente as colunas mínimas de timestamp.
- Grava resultados progressivamente no disco.
- Libera memória entre perfis com gc.collect().

Desenho
-------
Para cada usuária protegida:
1) lê o modelo vencedor do 10B;
2) recupera somente as predições OOF desse modelo;
3) anexa window_start_ms/window_end_ms;
4) agrega os scores dentro da MESMA session_id e do MESMO fold;
5) calcula média móvel retrospectiva dos últimos:
      5 s
      10 s
      15 s
      20 s
6) compara com janela única.

Como cada session_id pertence a uma única pessoa real, o rótulo
genuíno/impostor é constante dentro daquela sessão. Assim, a agregação
temporal não mistura identidades diferentes.

Métricas
--------
- EER
- FAR@EER
- FRR@EER
- ROC-AUC
- F1 da classe genuína no limiar EER
- Balanced Accuracy no limiar EER
- Accuracy no limiar EER

Entradas
--------
data/mouse_dataset/dataset_device_high_confidence.csv
data/mouse_authentication_1to1/predicoes_oof.csv
data/mouse_authentication_1to1/vencedores_por_perfil.csv

Saídas
------
data/mouse_authentication_temporal/
    resultados_temporais_por_fold.csv
    resumo_temporal_por_perfil.csv
    melhor_horizonte_por_perfil.csv
    decisoes_temporais.csv
    resultado_global_10c.csv
    manifesto_10c.json
"""

from __future__ import annotations

import gc
import json
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

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)

AUTH_DIR = (
    BASE_DIR
    / "data"
    / "mouse_authentication_1to1"
)

PREDICTIONS_PATH = (
    AUTH_DIR
    / "predicoes_oof.csv"
)

WINNERS_PATH = (
    AUTH_DIR
    / "vencedores_por_perfil.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "mouse_authentication_temporal"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIR
    / "resultados_temporais_por_fold.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "resumo_temporal_por_perfil.csv"
)

BEST_HORIZON_PATH = (
    OUTPUT_DIR
    / "melhor_horizonte_por_perfil.csv"
)

DECISIONS_PATH = (
    OUTPUT_DIR
    / "decisoes_temporais.csv"
)

GLOBAL_PATH = (
    OUTPUT_DIR
    / "resultado_global_10c.csv"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "manifesto_10c.json"
)

WINDOW_SIZE = 50
OVERLAP_PCT = 75

HORIZONS_SECONDS = [
    5,
    10,
    15,
    20,
]

# RAM conservadora: arquivo de predições lido em blocos pequenos.
CSV_CHUNK_SIZE = 20_000

# Para existir agregação real, exigimos ao menos 2 scores.
MIN_PREDICTIONS_PER_DECISION = 2

# Só começa a emitir decisão temporal depois de transcorrer
# o horizonte completo desde o início da sessão.
REQUIRE_FULL_HORIZON = True


# ============================================================
# UTILIDADES DE ARQUIVO
# ============================================================

def reset_outputs():
    for path in [
        FOLD_RESULTS_PATH,
        DECISIONS_PATH,
    ]:
        if path.exists():
            path.unlink()


def append_dataframe(
    df: pd.DataFrame,
    path: Path,
):
    if df.empty:
        return

    write_header = (
        not path.exists()
    )

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# MÉTRICAS
# ============================================================

def compute_eer(
    y_true: np.ndarray,
    scores_genuine: np.ndarray,
) -> dict:

    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores_genuine,
        dtype=np.float32,
    )

    if len(
        np.unique(
            y_true
        )
    ) < 2:
        return {
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "far_at_eer": float("nan"),
            "frr_at_eer": float("nan"),
        }

    fpr, tpr, thresholds = (
        roc_curve(
            y_true,
            scores,
            pos_label=1,
        )
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
        fpr[idx]
    )

    frr = float(
        fnr[idx]
    )

    eer = float(
        (
            far
            + frr
        )
        / 2.0
    )

    threshold = float(
        thresholds[idx]
    )

    return {
        "eer": eer,
        "eer_threshold": threshold,
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

    if len(
        y_true
    ) == 0:
        return {
            "n_decisions": 0,
            "n_genuine": 0,
            "n_impostor": 0,
            "eer": float("nan"),
            "eer_threshold": float("nan"),
            "far_at_eer": float("nan"),
            "frr_at_eer": float("nan"),
            "roc_auc": float("nan"),
            "f1_genuine_eer": float("nan"),
            "balanced_accuracy_eer": float("nan"),
            "accuracy_eer": float("nan"),
        }

    eer_info = compute_eer(
        y_true,
        scores,
    )

    if not np.isfinite(
        eer_info[
            "eer_threshold"
        ]
    ):
        pred = (
            scores
            >= 0.5
        ).astype(
            np.int8
        )
    else:
        pred = (
            scores
            >= eer_info[
                "eer_threshold"
            ]
        ).astype(
            np.int8
        )

    if len(
        np.unique(
            y_true
        )
    ) >= 2:
        auc = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )
    else:
        auc = float(
            "nan"
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
        "roc_auc": auc,
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
# DATASET MÍNIMO — BAIXO USO DE RAM
# ============================================================

def load_timestamp_lookup() -> pd.DataFrame:
    """
    Carrega SOMENTE as colunas necessárias para localizar temporalmente
    as janelas 50/75 de mouse.
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

    df = pd.read_csv(
        DATASET_PATH,
        usecols=usecols,
        low_memory=False,
        dtype={
            "window_id": "string",
            "session_id": "string",
            "window_size": "int16",
            "requested_overlap_pct": "int16",
            "device_context": "category",
            "window_start_ms": "int64",
            "window_end_ms": "int64",
        },
    )

    df = df[
        (
            df[
                "window_size"
            ]
            == WINDOW_SIZE
        )
        & (
            df[
                "requested_overlap_pct"
            ]
            == OVERLAP_PCT
        )
        & (
            df[
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

    if df.empty:
        raise RuntimeError(
            "Não encontrei timestamps para mouse 50/75."
        )

    duplicated = (
        df[
            "window_id"
        ]
        .duplicated(
            keep=False
        )
    )

    if duplicated.any():
        raise RuntimeError(
            "window_id duplicado no recorte 50/75."
        )

    return df


# ============================================================
# LEITURA CHUNKADA DAS PREDIÇÕES
# ============================================================

def load_winner_predictions_for_profile(
    protected_user: str,
    winner_candidate: str,
) -> pd.DataFrame:
    """
    Varre predicoes_oof.csv em chunks e mantém SOMENTE:
      protected_user == perfil atual
      candidate == vencedor do perfil

    Assim evitamos carregar o CSV inteiro na RAM.
    """

    usecols = [
        "protected_user",
        "candidate",
        "fold",
        "session_id",
        "window_id",
        "actual_user",
        "y_true_genuine",
        "score_genuine",
    ]

    parts = []

    for chunk in pd.read_csv(
        PREDICTIONS_PATH,
        usecols=usecols,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
        dtype={
            "protected_user": "string",
            "candidate": "string",
            "fold": "int8",
            "session_id": "string",
            "window_id": "string",
            "actual_user": "string",
            "y_true_genuine": "int8",
            "score_genuine": "float32",
        },
    ):

        mask = (
            chunk[
                "protected_user"
            ]
            .astype(str)
            .eq(
                protected_user
            )
            & chunk[
                "candidate"
            ]
            .astype(str)
            .eq(
                winner_candidate
            )
        )

        selected = chunk.loc[
            mask,
            [
                "protected_user",
                "candidate",
                "fold",
                "session_id",
                "window_id",
                "actual_user",
                "y_true_genuine",
                "score_genuine",
            ],
        ].copy()

        if not selected.empty:
            parts.append(
                selected
            )

        del chunk
        del selected
        gc.collect()

    if not parts:
        raise RuntimeError(
            f"Não encontrei predições para "
            f"{protected_user} / {winner_candidate}."
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts
    gc.collect()

    return result


# ============================================================
# AGREGAÇÃO TEMPORAL
# ============================================================

def aggregate_session(
    group: pd.DataFrame,
    horizon_seconds: int,
) -> pd.DataFrame:
    """
    Média móvel RETROSPECTIVA dos scores.

    Para a decisão na janela i:
    usa apenas scores de janelas com window_end_ms entre
    [tempo_atual - T, tempo_atual].

    Nenhuma informação futura é usada.
    """

    g = group.sort_values(
        [
            "window_end_ms",
            "window_start_ms",
            "window_id",
        ]
    ).reset_index(
        drop=True
    )

    times = pd.to_numeric(
        g[
            "window_end_ms"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.int64
    )

    scores = pd.to_numeric(
        g[
            "score_genuine"
        ],
        errors="raise",
    ).to_numpy(
        dtype=np.float32
    )

    labels = (
        g[
            "y_true_genuine"
        ]
        .astype(
            np.int8
        )
        .to_numpy()
    )

    if len(
        np.unique(
            labels
        )
    ) != 1:
        raise RuntimeError(
            "Uma session_id contém mais de um rótulo "
            "genuíno/impostor para o mesmo perfil protegido."
        )

    horizon_ms = int(
        horizon_seconds
        * 1000
    )

    # Acumulado em float64 para evitar perda numérica.
    cumulative = np.concatenate(
        [
            np.array(
                [0.0],
                dtype=np.float64,
            ),
            np.cumsum(
                scores,
                dtype=np.float64,
            ),
        ]
    )

    first_time = int(
        times[0]
    )

    left = 0
    rows = []

    for i in range(
        len(g)
    ):

        current_time = int(
            times[i]
        )

        while (
            left < i
            and (
                current_time
                - int(
                    times[left]
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
            < MIN_PREDICTIONS_PER_DECISION
        ):
            continue

        if REQUIRE_FULL_HORIZON:
            elapsed = (
                current_time
                - first_time
            )

            if elapsed < horizon_ms:
                continue

        sum_scores = (
            cumulative[
                i + 1
            ]
            - cumulative[
                left
            ]
        )

        mean_score = float(
            sum_scores
            / n_scores
        )

        rows.append(
            {
                "protected_user": str(
                    g.loc[
                        i,
                        "protected_user",
                    ]
                ),
                "candidate": str(
                    g.loc[
                        i,
                        "candidate",
                    ]
                ),
                "fold": int(
                    g.loc[
                        i,
                        "fold",
                    ]
                ),
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
                "horizon_s": int(
                    horizon_seconds
                ),
                "y_true_genuine": int(
                    labels[i]
                ),
                "score_single": float(
                    scores[i]
                ),
                "score_temporal": mean_score,
                "n_scores_aggregated": int(
                    n_scores
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_temporal_for_profile(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Processa horizonte por horizonte e sessão por sessão.
    Mantém uso de RAM pequeno.
    """

    parts = []

    grouped_keys = [
        "fold",
        "session_id",
    ]

    for horizon in HORIZONS_SECONDS:

        print(
            f"      horizonte {horizon}s..."
        )

        for _, group in predictions.groupby(
            grouped_keys,
            sort=True,
        ):

            result = aggregate_session(
                group,
                horizon_seconds=horizon,
            )

            if not result.empty:
                parts.append(
                    result
                )

            del result

        gc.collect()

    if not parts:
        raise RuntimeError(
            "Nenhuma decisão temporal gerada."
        )

    temporal = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts
    gc.collect()

    return temporal


# ============================================================
# AVALIAÇÃO POR FOLD
# ============================================================

def evaluate_single_window_by_fold(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for fold, group in predictions.groupby(
        "fold",
        sort=True,
    ):

        metrics = calculate_metrics(
            group[
                "y_true_genuine"
            ].to_numpy(),
            group[
                "score_genuine"
            ].to_numpy(),
        )

        rows.append(
            {
                "protected_user": str(
                    group[
                        "protected_user"
                    ].iloc[0]
                ),
                "candidate": str(
                    group[
                        "candidate"
                    ].iloc[0]
                ),
                "strategy": "single_window",
                "horizon_s": 0,
                "fold": int(
                    fold
                ),
                "mean_scores_per_decision": 1.0,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_temporal_by_fold(
    temporal: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        horizon,
        fold,
    ), group in temporal.groupby(
        [
            "horizon_s",
            "fold",
        ],
        sort=True,
    ):

        metrics = calculate_metrics(
            group[
                "y_true_genuine"
            ].to_numpy(),
            group[
                "score_temporal"
            ].to_numpy(),
        )

        rows.append(
            {
                "protected_user": str(
                    group[
                        "protected_user"
                    ].iloc[0]
                ),
                "candidate": str(
                    group[
                        "candidate"
                    ].iloc[0]
                ),
                "strategy": "temporal_mean",
                "horizon_s": int(
                    horizon
                ),
                "fold": int(
                    fold
                ),
                "mean_scores_per_decision": float(
                    group[
                        "n_scores_aggregated"
                    ].mean()
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RESUMOS
# ============================================================

def summarize_fold_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "eer",
        "far_at_eer",
        "frr_at_eer",
        "roc_auc",
        "f1_genuine_eer",
        "balanced_accuracy_eer",
        "accuracy_eer",
        "eer_threshold",
        "mean_scores_per_decision",
        "n_decisions",
    ]

    rows = []

    grouped = fold_results.groupby(
        [
            "protected_user",
            "candidate",
            "strategy",
            "horizon_s",
        ],
        sort=True,
        dropna=False,
    )

    for keys, group in grouped:

        (
            protected_user,
            candidate,
            strategy,
            horizon_s,
        ) = keys

        row = {
            "protected_user": protected_user,
            "candidate": candidate,
            "strategy": strategy,
            "horizon_s": int(
                horizon_s
            ),
            "n_folds": int(
                len(
                    group
                )
            ),
        }

        for metric in metrics:
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

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    return summary


def choose_best_horizon(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    winners = []

    for protected_user, group in summary.groupby(
        "protected_user",
        sort=True,
    ):

        # inclui também a janela única; se agregação piorar,
        # a janela única pode vencer.
        ranked = group.sort_values(
            [
                "eer_mean",
                "roc_auc_mean",
                "balanced_accuracy_eer_mean",
                "f1_genuine_eer_mean",
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
            ranked.iloc[0].copy()
        )

    return pd.DataFrame(
        winners
    ).reset_index(
        drop=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "="
        * 104
    )

    print(
        "CYBELL-J — EXPERIMENTO 10C — "
        "AGREGACAO TEMPORAL DA AUTENTICACAO 1:1"
    )

    print(
        "="
        * 104
    )

    print(
        "\nPerfil de recursos:"
        "\n- sem treinamento"
        "\n- sem PyCaret"
        "\n- sem GPU"
        f"\n- leitura de predições em chunks de {CSV_CHUNK_SIZE:,}"
        "\n- uma usuária por vez"
        "\n- gravação incremental"
        "\n- liberação de memória entre perfis"
    )

    for required_path in [
        DATASET_PATH,
        PREDICTIONS_PATH,
        WINNERS_PATH,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado:\n"
                f"{required_path}"
            )

    reset_outputs()

    print(
        "\n1) Carregando apenas timestamps mínimos "
        "do dataset..."
    )

    timestamps = load_timestamp_lookup()

    print(
        f"   Janelas timestamp carregadas: "
        f"{len(timestamps):,}"
        .replace(",", ".")
    )

    print(
        "\n2) Carregando vencedores do 10B..."
    )

    winners = pd.read_csv(
        WINNERS_PATH,
        low_memory=False,
        dtype={
            "protected_user": "string",
            "candidate": "string",
        },
    )

    needed = {
        "protected_user",
        "candidate",
    }

    missing = sorted(
        needed
        - set(
            winners.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Colunas ausentes em vencedores_por_perfil.csv: "
            f"{missing}"
        )

    profile_summaries = []

    print(
        "\n"
        + "="
        * 104
    )

    print(
        "3) PROCESSANDO PERFIS"
    )

    print(
        "="
        * 104
    )

    for index, winner in winners.iterrows():

        protected_user = str(
            winner[
                "protected_user"
            ]
        )

        candidate = str(
            winner[
                "candidate"
            ]
        )

        print(
            "\n"
            + "-"
            * 104
        )

        print(
            f"[{index + 1}/{len(winners)}] "
            f"{protected_user} | {candidate}"
        )

        print(
            "-"
            * 104
        )

        print(
            "   A) Lendo somente as predições "
            "deste perfil/modelo..."
        )

        profile_pred = (
            load_winner_predictions_for_profile(
                protected_user=protected_user,
                winner_candidate=candidate,
            )
        )

        print(
            f"      predições: "
            f"{len(profile_pred):,}"
            .replace(",", ".")
        )

        print(
            "   B) Anexando timestamps..."
        )

        profile_pred = profile_pred.merge(
            timestamps,
            on=[
                "window_id",
                "session_id",
            ],
            how="left",
            validate="many_to_one",
        )

        missing_time = (
            profile_pred[
                "window_end_ms"
            ]
            .isna()
        )

        if missing_time.any():
            examples = (
                profile_pred.loc[
                    missing_time,
                    "window_id",
                ]
                .head(
                    10
                )
                .tolist()
            )

            raise RuntimeError(
                "Algumas janelas não receberam timestamp. "
                f"Exemplos: {examples}"
            )

        # Garante tipos compactos após merge.
        profile_pred[
            "score_genuine"
        ] = pd.to_numeric(
            profile_pred[
                "score_genuine"
            ],
            errors="raise",
        ).astype(
            "float32"
        )

        profile_pred[
            "y_true_genuine"
        ] = profile_pred[
            "y_true_genuine"
        ].astype(
            "int8"
        )

        profile_pred[
            "fold"
        ] = profile_pred[
            "fold"
        ].astype(
            "int8"
        )

        print(
            "   C) Avaliando janela única..."
        )

        single_folds = (
            evaluate_single_window_by_fold(
                profile_pred
            )
        )

        print(
            "   D) Agregando 5/10/15/20s..."
        )

        temporal = (
            build_temporal_for_profile(
                profile_pred
            )
        )

        # grava decisões temporais imediatamente
        append_dataframe(
            temporal,
            DECISIONS_PATH,
        )

        temporal_folds = (
            evaluate_temporal_by_fold(
                temporal
            )
        )

        profile_folds = pd.concat(
            [
                single_folds,
                temporal_folds,
            ],
            ignore_index=True,
        )

        append_dataframe(
            profile_folds,
            FOLD_RESULTS_PATH,
        )

        profile_summary = (
            summarize_fold_results(
                profile_folds
            )
        )

        profile_summaries.append(
            profile_summary
        )

        print(
            "\n   Resultado:"
        )

        print(
            profile_summary[
                [
                    "strategy",
                    "horizon_s",
                    "eer_mean",
                    "roc_auc_mean",
                    "f1_genuine_eer_mean",
                    "balanced_accuracy_eer_mean",
                    "mean_scores_per_decision_mean",
                ]
            ]
            .sort_values(
                "horizon_s"
            )
            .to_string(
                index=False
            )
        )

        best_profile = (
            choose_best_horizon(
                profile_summary
            )
            .iloc[0]
        )

        print(
            "\n   MELHOR:"
        )

        print(
            f"      Horizonte: "
            f"{int(best_profile['horizon_s'])} s"
        )

        print(
            f"      EER: "
            f"{best_profile['eer_mean']:.4f}"
        )

        print(
            f"      AUC: "
            f"{best_profile['roc_auc_mean']:.4f}"
        )

        print(
            f"      F1@EER: "
            f"{best_profile['f1_genuine_eer_mean']:.4f}"
        )

        del profile_pred
        del single_folds
        del temporal
        del temporal_folds
        del profile_folds
        del profile_summary
        del best_profile

        gc.collect()

        print(
            "   Memória liberada."
        )

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    summary = pd.concat(
        profile_summaries,
        ignore_index=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    best_horizons = (
        choose_best_horizon(
            summary
        )
    )

    best_horizons.to_csv(
        BEST_HORIZON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "="
        * 104
    )

    print(
        "4) MELHOR HORIZONTE POR PERFIL"
    )

    print(
        "="
        * 104
    )

    print(
        best_horizons[
            [
                "protected_user",
                "candidate",
                "strategy",
                "horizon_s",
                "eer_mean",
                "far_at_eer_mean",
                "frr_at_eer_mean",
                "roc_auc_mean",
                "f1_genuine_eer_mean",
                "balanced_accuracy_eer_mean",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # COMPARAÇÃO GLOBAL DOS HORIZONTES
    # ========================================================

    global_rows = []

    for (
        strategy,
        horizon,
    ), group in summary.groupby(
        [
            "strategy",
            "horizon_s",
        ],
        sort=True,
    ):

        global_rows.append(
            {
                "strategy": strategy,
                "horizon_s": int(
                    horizon
                ),
                "profiles": int(
                    group[
                        "protected_user"
                    ].nunique()
                ),
                "eer_mean_profiles": float(
                    group[
                        "eer_mean"
                    ].mean()
                ),
                "eer_median_profiles": float(
                    group[
                        "eer_mean"
                    ].median()
                ),
                "auc_mean_profiles": float(
                    group[
                        "roc_auc_mean"
                    ].mean()
                ),
                "f1_eer_mean_profiles": float(
                    group[
                        "f1_genuine_eer_mean"
                    ].mean()
                ),
                "balanced_accuracy_eer_mean_profiles": float(
                    group[
                        "balanced_accuracy_eer_mean"
                    ].mean()
                ),
            }
        )

    global_df = pd.DataFrame(
        global_rows
    )

    global_df = global_df.sort_values(
        [
            "eer_mean_profiles",
            "auc_mean_profiles",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    global_df.to_csv(
        GLOBAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "="
        * 104
    )

    print(
        "5) RESULTADO GLOBAL 10C"
    )

    print(
        "="
        * 104
    )

    print(
        global_df.to_string(
            index=False
        )
    )

    best_global = (
        global_df.iloc[0]
    )

    print(
        "\nMELHOR CONFIGURACAO GLOBAL:"
    )

    print(
        f"   Estratégia: "
        f"{best_global['strategy']}"
    )

    print(
        f"   Horizonte: "
        f"{int(best_global['horizon_s'])} s"
    )

    print(
        f"   EER médio entre perfis: "
        f"{best_global['eer_mean_profiles']:.4f}"
    )

    print(
        f"   AUC média entre perfis: "
        f"{best_global['auc_mean_profiles']:.4f}"
    )

    print(
        f"   F1@EER médio entre perfis: "
        f"{best_global['f1_eer_mean_profiles']:.4f}"
    )

    # ========================================================
    # MANIFESTO
    # ========================================================

    manifest = {
        "projeto": "CYBELL-J",
        "fase": (
            "Experimento 10C - agregacao temporal "
            "da autenticacao continua 1:1"
        ),
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": DATASET_PATH.name,
        "source_predictions": PREDICTIONS_PATH.name,
        "source_winners": WINNERS_PATH.name,
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP_PCT,
        "horizons_seconds": HORIZONS_SECONDS,
        "resource_profile": {
            "training": False,
            "pycaret": False,
            "gpu": False,
            "csv_chunk_size": CSV_CHUNK_SIZE,
            "profiles_processed_one_at_a_time": True,
            "incremental_writes": True,
            "gc_between_profiles": True,
            "minimal_dataset_columns": True,
        },
        "aggregation": {
            "method": "trailing_mean_of_genuine_scores",
            "same_session_only": True,
            "same_fold_only": True,
            "uses_future_predictions": False,
            "require_full_horizon": REQUIRE_FULL_HORIZON,
            "minimum_scores": MIN_PREDICTIONS_PER_DECISION,
        },
        "best_global": {
            "strategy": str(
                best_global[
                    "strategy"
                ]
            ),
            "horizon_s": int(
                best_global[
                    "horizon_s"
                ]
            ),
            "eer_mean_profiles": float(
                best_global[
                    "eer_mean_profiles"
                ]
            ),
            "auc_mean_profiles": float(
                best_global[
                    "auc_mean_profiles"
                ]
            ),
            "f1_eer_mean_profiles": float(
                best_global[
                    "f1_eer_mean_profiles"
                ]
            ),
        },
        "methodological_notes": [
            "Nenhum modelo e retreinado no 10C.",
            "Somente as predicoes OOF do candidato vencedor de cada perfil no 10B sao usadas.",
            "A agregacao e retrospectiva e ocorre apenas dentro da mesma session_id.",
            "Como cada session_id pertence a uma unica pessoa real, o rotulo genuino/impostor e constante dentro da sessao.",
            "O EER e recalculado separadamente por fold para cada horizonte antes da media entre folds.",
            "O script foi desenhado para baixo consumo de RAM: leitura chunkada, tipos compactos, uma usuaria por vez e gravacao incremental.",
        ],
        "outputs": {
            "fold_results": FOLD_RESULTS_PATH.name,
            "summary": SUMMARY_PATH.name,
            "best_horizon_per_profile": BEST_HORIZON_PATH.name,
            "temporal_decisions": DECISIONS_PATH.name,
            "global_result": GLOBAL_PATH.name,
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

    print(
        "\n"
        + "="
        * 104
    )

    print(
        "EXPERIMENTO 10C CONCLUIDO"
    )

    print(
        "="
        * 104
    )

    print(
        f"\nResultados por fold:\n"
        f"{FOLD_RESULTS_PATH}"
    )

    print(
        f"\nResumo temporal:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nMelhor horizonte por perfil:\n"
        f"{BEST_HORIZON_PATH}"
    )

    print(
        f"\nDecisoes temporais:\n"
        f"{DECISIONS_PATH}"
    )

    print(
        f"\nResultado global:\n"
        f"{GLOBAL_PATH}"
    )

    print(
        f"\nManifesto:\n"
        f"{MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()