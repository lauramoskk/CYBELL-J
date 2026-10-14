"""
11A_perfil_unico_vs_subperfil_mouse_bianca.py

CYBELL-J — Experimento 11A
COMPARAÇÃO: PERFIL ÚNICO x SUBPERFIL MOUSE

Pergunta
--------
"Para uma usuária que utiliza mouse e trackpad, misturar os dois contextos
no mesmo perfil comportamental piora a autenticação quando ela está usando
mouse? Um subperfil específico de mouse melhora o EER?"

Por que Bianca?
---------------
Bianca é a única participante com volume substancial de mouse E trackpad.
Por isso, esta é a única comparação de subperfil que pode ser feita com
quantidade razoável de dados nesta coleta.

O que ESTE experimento valida
-----------------------------
Avaliação no contexto MOUSE:

A) PERFIL_UNICO
   Genuíno no treino:
       Bianca mouse + Bianca trackpad
   Impostores no treino:
       mouse das outras usuárias
   Teste:
       somente mouse, usando as mesmas sessões externas

B) SUBPERFIL_MOUSE
   Genuíno no treino:
       somente Bianca mouse
   Impostores no treino:
       mouse das outras usuárias
   Teste:
       exatamente o mesmo teste mouse usado em A

Assim, a principal diferença entre A e B é:
    misturar ou não o comportamento de trackpad no perfil genuíno.

IMPORTANTE
----------
- O teste é SOMENTE em mouse.
- Não treinamos ainda um "subperfil trackpad" de autenticação porque não
  há quantidade suficiente de impostores em trackpad de várias usuárias.
- Portanto, este experimento NÃO demonstra autenticação universal em trackpad.
- Nenhuma session_id do teste aparece no treino.
- As 3 sessões mouse da Bianca são distribuídas entre os 3 folds externos.

PyCaret
-------
O PyCaret NÃO foi deixado de lado.

Dentro de CADA fold externo:
1) PyCaret recebe SOMENTE o conjunto de treino daquele fold;
2) compara LR / Random Forest / Extra Trees;
3) usa validação interna por sessão;
4) escolhe o melhor por AUC;
5) o modelo escolhido é avaliado na sessão externa nunca vista.

Também avaliamos:
    XGBoost_GPU_Light

Uso de recursos
---------------
Perfil conservador:
- CPU: 2 threads
- PyCaret: CPU, sem GPU
- XGBoost: GPU apenas em bursts curtos
- XGBoost leve: 140 árvores, depth 4, max_bin 128
- treino impostor:genuíno <= 2:1
- features convertidas para float32
- dataset lido em chunks
- resultados gravados incrementalmente
- gc.collect() após cada fold
- não acumula grandes matrizes de predição na RAM

Configuração de janela
----------------------
50 eventos / 75% overlap
(melhor configuração encontrada no Experimento 10A)

Saídas
------
data/profile_vs_mouse_subprofile/
    resultados_folds.csv
    resumo_estrategias_modelos.csv
    vencedores_por_estrategia.csv
    comparacao_final.csv
    predicoes_oof.csv
    leaderboards_pycaret/
    manifesto_11a.json
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import pycaret
    try:
        from pycaret.tasks import ClassificationExperiment
    except ImportError:
        from pycaret.classification import ClassificationExperiment
except ImportError as exc:
    raise SystemExit(
        "\nERRO: PyCaret não está disponível neste ambiente.\n"
        "Ative o ambiente correto:\n"
        "    deactivate\n"
        "    .\\.venv_pycaret\\Scripts\\Activate.ps1\n"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "\nERRO: xgboost não está instalado.\n"
        "Execute:\n"
        "    python -m pip install xgboost\n"
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
    / "profile_vs_mouse_subprofile"
)

LEADERBOARD_DIR = (
    OUTPUT_DIR
    / "leaderboards_pycaret"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LEADERBOARD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FOLDS_PATH = (
    OUTPUT_DIR
    / "resultados_folds.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "resumo_estrategias_modelos.csv"
)

WINNERS_PATH = (
    OUTPUT_DIR
    / "vencedores_por_estrategia.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIR
    / "comparacao_final.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "predicoes_oof.csv"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "manifesto_11a.json"
)

PROTECTED_USER = "bianca"

WINDOW_SIZE = 50
OVERLAP_PCT = 75

N_OUTER_FOLDS = 3
N_INNER_FOLDS = 2

RANDOM_STATE = 42

CPU_JOBS = 2

PYCARET_USE_GPU = False
PYCARET_INCLUDE = [
    "lr",
    "rf",
    "et",
]

IMPOSTOR_TO_GENUINE_TRAIN_RATIO = 2.0

XGB_N_ESTIMATORS = 140
XGB_MAX_DEPTH = 4
XGB_MAX_BIN = 128
XGB_LEARNING_RATE = 0.06
XGB_SUBSAMPLE = 0.80
XGB_COLSAMPLE = 0.80

CSV_CHUNK_SIZE = 50_000

STRATEGIES = [
    "PERFIL_UNICO",
    "SUBPERFIL_MOUSE",
]

os.environ.setdefault(
    "OMP_NUM_THREADS",
    str(CPU_JOBS),
)
os.environ.setdefault(
    "MKL_NUM_THREADS",
    str(CPU_JOBS),
)
os.environ.setdefault(
    "OPENBLAS_NUM_THREADS",
    str(CPU_JOBS),
)
os.environ.setdefault(
    "NUMEXPR_NUM_THREADS",
    str(CPU_JOBS),
)


# ============================================================
# METADADOS / FEATURES
# ============================================================

METADATA_COLUMNS = {
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
    "device_mouse_pct",
    "device_trackpad_pct",
    "device_context",
    "device_confidence",
    "device_high_confidence",
    "device_label_source",
    "target_user_device",
}


def select_features(
    df: pd.DataFrame,
) -> list[str]:

    features = []

    for column in df.columns:

        if column in METADATA_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            features.append(
                column
            )

    if not features:
        raise RuntimeError(
            "Nenhuma feature numérica encontrada."
        )

    return features


# ============================================================
# CARREGAMENTO CHUNKADO
# ============================================================

def load_configuration_low_ram() -> pd.DataFrame:
    """
    Lê o CSV em chunks e mantém somente:
        window_size=50
        overlap=75
        device_context mouse ou trackpad
    """

    parts = []

    for chunk in pd.read_csv(
        DATASET_PATH,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ):

        selected = chunk[
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
                .isin(
                    [
                        "mouse",
                        "trackpad",
                    ]
                )
            )
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
            "Nenhuma janela 50/75 encontrada."
        )

    df = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts
    gc.collect()

    return df


def compact_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:

    for column in features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).astype(
            "float32"
        )

    return df


# ============================================================
# FOLDS EXTERNOS — BASEADOS NO MOUSE
# ============================================================

def stable_seed(
    text: str,
) -> int:

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        RANDOM_STATE
        + int(
            digest[:8],
            16,
        )
    ) % (
        2**32 - 1
    )


def build_mouse_outer_folds(
    mouse_df: pd.DataFrame,
) -> list[dict]:
    """
    Cada session_id mouse aparece em TESTE exatamente uma vez.

    Para cada usuária, suas sessões são distribuídas round-robin.
    Bianca tem 3 sessões mouse, então cada fold testa uma delas.
    """

    users = sorted(
        mouse_df[
            "user"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    fold_test_sessions = [
        []
        for _ in range(
            N_OUTER_FOLDS
        )
    ]

    for user in users:

        sessions = sorted(
            mouse_df.loc[
                mouse_df[
                    "user"
                ]
                .astype(str)
                .eq(
                    user
                ),
                "session_id",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(
            sessions
        ) < N_OUTER_FOLDS:
            raise RuntimeError(
                f"{user} possui apenas "
                f"{len(sessions)} sessões mouse."
            )

        rng = np.random.default_rng(
            stable_seed(
                f"outer|{user}"
            )
        )

        rng.shuffle(
            sessions
        )

        for idx, session in enumerate(
            sessions
        ):
            fold_test_sessions[
                idx
                % N_OUTER_FOLDS
            ].append(
                session
            )

    all_mouse_sessions = set(
        mouse_df[
            "session_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    folds = []

    seen_test = []

    for fold_idx in range(
        N_OUTER_FOLDS
    ):

        test_sessions = sorted(
            set(
                fold_test_sessions[
                    fold_idx
                ]
            )
        )

        train_mouse_sessions = sorted(
            all_mouse_sessions
            - set(
                test_sessions
            )
        )

        folds.append(
            {
                "fold": fold_idx + 1,
                "test_sessions": test_sessions,
                "train_mouse_sessions": train_mouse_sessions,
            }
        )

        seen_test.extend(
            test_sessions
        )

    if len(
        seen_test
    ) != len(
        set(
            seen_test
        )
    ):
        raise RuntimeError(
            "Uma sessão mouse apareceu em teste mais de uma vez."
        )

    return folds


# ============================================================
# CONJUNTOS DE TREINO E TESTE
# ============================================================

def build_train_test_for_strategy(
    all_df: pd.DataFrame,
    mouse_df: pd.DataFrame,
    fold_info: dict,
    strategy: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    test_sessions = set(
        fold_info[
            "test_sessions"
        ]
    )

    # TESTE IDÊNTICO para as duas estratégias:
    # somente mouse, nas sessões externas.
    test = mouse_df[
        mouse_df[
            "session_id"
        ]
        .astype(str)
        .isin(
            test_sessions
        )
    ].copy()

    # Impostores de treino: sempre somente mouse.
    impostor_train = mouse_df[
        (
            ~mouse_df[
                "user"
            ]
            .astype(str)
            .eq(
                PROTECTED_USER
            )
        )
        & (
            ~mouse_df[
                "session_id"
            ]
            .astype(str)
            .isin(
                test_sessions
            )
        )
    ].copy()

    if strategy == "SUBPERFIL_MOUSE":

        genuine_train = mouse_df[
            (
                mouse_df[
                    "user"
                ]
                .astype(str)
                .eq(
                    PROTECTED_USER
                )
            )
            & (
                ~mouse_df[
                    "session_id"
                ]
                .astype(str)
                .isin(
                    test_sessions
                )
            )
        ].copy()

    elif strategy == "PERFIL_UNICO":

        # Perfil único:
        # usa mouse + trackpad genuínos da Bianca,
        # mas NUNCA usa qualquer janela da session_id externa.
        genuine_train = all_df[
            (
                all_df[
                    "user"
                ]
                .astype(str)
                .eq(
                    PROTECTED_USER
                )
            )
            & (
                ~all_df[
                    "session_id"
                ]
                .astype(str)
                .isin(
                    test_sessions
                )
            )
        ].copy()

    else:
        raise RuntimeError(
            f"Estratégia desconhecida: "
            f"{strategy}"
        )

    train = pd.concat(
        [
            genuine_train,
            impostor_train,
        ],
        ignore_index=False,
    )

    # Segurança
    leak = (
        set(
            train[
                "session_id"
            ]
            .astype(str)
            .unique()
        )
        & set(
            test[
                "session_id"
            ]
            .astype(str)
            .unique()
        )
    )

    if leak:
        raise RuntimeError(
            f"Vazamento de sessão em "
            f"{strategy}: {sorted(leak)}"
        )

    if test.empty:
        raise RuntimeError(
            "Conjunto de teste vazio."
        )

    return (
        train,
        test,
    )


# ============================================================
# SUBAMOSTRAGEM DE IMPOSTORES
# ============================================================

def sample_impostors(
    train: pd.DataFrame,
    protected_user: str,
) -> pd.DataFrame:

    genuine = train[
        train[
            "user"
        ]
        .astype(str)
        .eq(
            protected_user
        )
    ].copy()

    impostors = train[
        ~train[
            "user"
        ]
        .astype(str)
        .eq(
            protected_user
        )
    ].copy()

    if genuine.empty:
        raise RuntimeError(
            "Sem genuínos no treino."
        )

    target_impostors = int(
        round(
            len(
                genuine
            )
            * IMPOSTOR_TO_GENUINE_TRAIN_RATIO
        )
    )

    if len(
        impostors
    ) <= target_impostors:
        sampled_impostors = (
            impostors
        )
    else:

        impostor_users = sorted(
            impostors[
                "user"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        parts = []

        remaining = (
            target_impostors
        )

        for idx, impostor_user in enumerate(
            impostor_users
        ):

            pool = impostors[
                impostors[
                    "user"
                ]
                .astype(str)
                .eq(
                    impostor_user
                )
            ]

            users_left = (
                len(
                    impostor_users
                )
                - idx
            )

            desired = min(
                len(
                    pool
                ),
                max(
                    1,
                    remaining
                    // users_left,
                ),
            )

            sampled = pool.sample(
                n=desired,
                replace=False,
                random_state=stable_seed(
                    f"sample|{protected_user}|"
                    f"{impostor_user}|"
                    f"{len(train)}"
                ),
            )

            parts.append(
                sampled
            )

            remaining -= len(
                sampled
            )

        sampled_impostors = pd.concat(
            parts,
            ignore_index=False,
        )

    balanced = pd.concat(
        [
            genuine,
            sampled_impostors,
        ],
        ignore_index=False,
    )

    balanced = balanced.sample(
        frac=1.0,
        random_state=stable_seed(
            f"shuffle|{len(train)}"
        ),
    )

    return balanced


# ============================================================
# CV INTERNO DO PYCARET
# ============================================================

class SessionCV(
    BaseCrossValidator
):
    def __init__(
        self,
        index_to_session: dict,
        fold_defs: list[dict],
    ):
        self.index_to_session = (
            index_to_session
        )

        self.fold_defs = (
            fold_defs
        )

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None,
    ):
        return len(
            self.fold_defs
        )

    def split(
        self,
        X,
        y=None,
        groups=None,
    ):
        session_values = np.array(
            [
                self.index_to_session.get(
                    idx
                )
                for idx in X.index
            ],
            dtype=object,
        )

        for fold in self.fold_defs:

            test_sessions = set(
                fold[
                    "test_sessions"
                ]
            )

            test_mask = np.array(
                [
                    session
                    in test_sessions
                    for session
                    in session_values
                ],
                dtype=bool,
            )

            train_idx = np.flatnonzero(
                ~test_mask
            )

            test_idx = np.flatnonzero(
                test_mask
            )

            if (
                len(
                    train_idx
                )
                == 0
                or len(
                    test_idx
                )
                == 0
            ):
                continue

            if y is not None:
                y_arr = np.asarray(
                    y
                )

                if len(
                    np.unique(
                        y_arr[
                            train_idx
                        ]
                    )
                ) < 2:
                    continue

                if len(
                    np.unique(
                        y_arr[
                            test_idx
                        ]
                    )
                ) < 2:
                    continue

            yield (
                train_idx,
                test_idx,
            )


def build_inner_session_folds(
    train_balanced: pd.DataFrame,
) -> list[dict]:
    """
    Cria 2 folds internos por sessão.

    Distribui separadamente:
    - sessões genuínas da Bianca
    - sessões dos impostores
    """

    session_labels = (
        train_balanced[
            [
                "session_id",
                "user",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    session_labels[
        "target_bin"
    ] = (
        session_labels[
            "user"
        ]
        .astype(str)
        .eq(
            PROTECTED_USER
        )
        .astype(int)
    )

    fold_sessions = [
        []
        for _ in range(
            N_INNER_FOLDS
        )
    ]

    for label in [
        0,
        1,
    ]:

        sessions = sorted(
            session_labels.loc[
                session_labels[
                    "target_bin"
                ]
                == label,
                "session_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(
            sessions
        ) < N_INNER_FOLDS:
            raise RuntimeError(
                "Poucas sessões para CV interno "
                f"na classe {label}: "
                f"{len(sessions)}"
            )

        rng = np.random.default_rng(
            stable_seed(
                f"inner|{label}|"
                f"{len(train_balanced)}"
            )
        )

        rng.shuffle(
            sessions
        )

        for idx, session in enumerate(
            sessions
        ):
            fold_sessions[
                idx
                % N_INNER_FOLDS
            ].append(
                session
            )

    return [
        {
            "test_sessions": sorted(
                set(
                    fold_sessions[
                        idx
                    ]
                )
            )
        }
        for idx in range(
            N_INNER_FOLDS
        )
    ]


def pycaret_select_on_training_only(
    train_balanced: pd.DataFrame,
    features: list[str],
    strategy: str,
    outer_fold: int,
):
    """
    PyCaret vê SOMENTE o treino do fold externo.
    A sessão externa nunca participa da escolha do algoritmo.
    """

    work = train_balanced[
        features
        + [
            "user",
            "session_id",
        ]
    ].copy()

    work[
        "target_bin"
    ] = (
        work[
            "user"
        ]
        .astype(str)
        .eq(
            PROTECTED_USER
        )
        .astype(int)
    )

    inner_folds = (
        build_inner_session_folds(
            work
        )
    )

    index_to_session = (
        work[
            "session_id"
        ]
        .astype(str)
        .to_dict()
    )

    cv = SessionCV(
        index_to_session=index_to_session,
        fold_defs=inner_folds,
    )

    py_data = work[
        features
        + [
            "target_bin"
        ]
    ].copy()

    exp = ClassificationExperiment(
        target="target_bin",
        session_id=RANDOM_STATE,
        train_size=0.95,
        fold=N_INNER_FOLDS,
        preprocess=True,
        normalize=False,
        transformation=False,
        remove_outliers=False,
        feature_selection=False,
        n_jobs=CPU_JOBS,
        use_gpu=PYCARET_USE_GPU,
        verbose=False,
    )

    exp.fit(
        py_data
    )

    result = exp.compare_models(
        fold=cv,
        include=PYCARET_INCLUDE,
        sort="AUC",
        n_select=1,
        turbo=True,
        errors="ignore",
        round=4,
        verbose=False,
    )

    leaderboard = (
        result.leaderboard.copy()
    )

    leaderboard.to_csv(
        LEADERBOARD_DIR
        / (
            f"{strategy}"
            f"_fold{outer_fold}"
            f"_leaderboard.csv"
        ),
        index=False,
        encoding="utf-8-sig",
    )

    if not result.models:
        raise RuntimeError(
            "PyCaret não retornou modelo."
        )

    pipeline = result.models[0]

    model_id = (
        result.ranked_ids[0]
        if result.ranked_ids
        else "best"
    )

    del exp
    del result
    del py_data
    del work

    gc.collect()

    return (
        pipeline,
        str(
            model_id
        ),
    )


# ============================================================
# XGBOOST LEVE
# ============================================================

def gpu_available() -> bool:

    try:

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        X = rng.random(
            (
                128,
                8,
            ),
            dtype=np.float32,
        )

        y = rng.integers(
            0,
            2,
            size=128,
        )

        m = XGBClassifier(
            n_estimators=2,
            max_depth=2,
            max_bin=64,
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            verbosity=0,
        )

        m.fit(
            X,
            y,
        )

        del m
        del X
        del y

        gc.collect()

        return True

    except Exception:
        return False


def build_xgb_light(
    y_train: np.ndarray,
    use_gpu: bool,
) -> Pipeline:

    positives = int(
        np.sum(
            y_train
            == 1
        )
    )

    negatives = int(
        np.sum(
            y_train
            == 0
        )
    )

    scale_pos_weight = (
        negatives
        / positives
        if positives > 0
        else 1.0
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=XGB_N_ESTIMATORS,
                    learning_rate=XGB_LEARNING_RATE,
                    max_depth=XGB_MAX_DEPTH,
                    min_child_weight=2,
                    subsample=XGB_SUBSAMPLE,
                    colsample_bytree=XGB_COLSAMPLE,
                    max_bin=XGB_MAX_BIN,
                    tree_method="hist",
                    device=(
                        "cuda"
                        if use_gpu
                        else "cpu"
                    ),
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    n_jobs=CPU_JOBS,
                    verbosity=0,
                ),
            ),
        ]
    )


# ============================================================
# MÉTRICAS
# ============================================================

def get_score(
    model,
    X,
) -> np.ndarray:

    if hasattr(
        model,
        "predict_proba",
    ):

        proba = np.asarray(
            model.predict_proba(
                X
            )
        )

        classes = list(
            getattr(
                model,
                "classes_",
                [
                    0,
                    1,
                ],
            )
        )

        idx = (
            classes.index(
                1
            )
            if 1 in classes
            else 1
        )

        return proba[
            :,
            idx
        ].astype(
            float
        )

    if hasattr(
        model,
        "decision_function",
    ):
        return np.asarray(
            model.decision_function(
                X
            ),
            dtype=float,
        )

    return np.asarray(
        model.predict(
            X
        ),
        dtype=float,
    )


def compute_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

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
# OUTPUT INCREMENTAL
# ============================================================

def reset_outputs():

    for path in [
        FOLDS_PATH,
        PREDICTIONS_PATH,
    ]:

        if path.exists():
            path.unlink()


def append_df(
    df: pd.DataFrame,
    path: Path,
):

    df.to_csv(
        path,
        mode="a",
        header=(
            not path.exists()
        ),
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "="
        * 106
    )

    print(
        "CYBELL-J — EXPERIMENTO 11A — "
        "PERFIL UNICO x SUBPERFIL MOUSE — BIANCA"
    )

    print(
        "="
        * 106
    )

    print(
        f"\nPyCaret: {pycaret.__version__}"
        f"\nCPU: {CPU_JOBS} threads"
        f"\nPyCaret GPU: {PYCARET_USE_GPU}"
        "\nXGBoost GPU: leve e apenas em bursts"
        "\nRAM: leitura chunkada + float32 + liberação por fold"
    )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset não encontrado:\n"
            f"{DATASET_PATH}"
        )

    print(
        "\n1) Testando CUDA..."
    )

    use_gpu = gpu_available()

    print(
        "   CUDA XGBoost: "
        + (
            "disponível"
            if use_gpu
            else "indisponível; fallback CPU"
        )
    )

    print(
        "\n2) Carregando apenas "
        "a configuração 50/75 em chunks..."
    )

    all_df = (
        load_configuration_low_ram()
    )

    features = select_features(
        all_df
    )

    all_df = compact_features(
        all_df,
        features,
    )

    mouse_df = all_df[
        all_df[
            "device_context"
        ]
        .astype(str)
        .eq(
            "mouse"
        )
    ].copy()

    print(
        f"   Total 50/75 mouse+trackpad: "
        f"{len(all_df):,}"
        .replace(
            ",",
            ".",
        )
    )

    print(
        f"   Mouse: "
        f"{len(mouse_df):,}"
        .replace(
            ",",
            ".",
        )
    )

    bianca_counts = (
        all_df[
            all_df[
                "user"
            ]
            .astype(str)
            .eq(
                PROTECTED_USER
            )
        ]
        .groupby(
            "device_context"
        )
        .size()
    )

    print(
        "\n   Bianca por contexto:"
    )

    print(
        bianca_counts.to_string()
    )

    print(
        f"\n   Features: "
        f"{len(features)}"
    )

    outer_folds = (
        build_mouse_outer_folds(
            mouse_df
        )
    )

    print(
        "\n3) Folds externos:"
    )

    for fold in outer_folds:

        bianca_test_sessions = sorted(
            set(
                mouse_df.loc[
                    (
                        mouse_df[
                            "user"
                        ]
                        .astype(str)
                        .eq(
                            PROTECTED_USER
                        )
                    )
                    & (
                        mouse_df[
                            "session_id"
                        ]
                        .astype(str)
                        .isin(
                            fold[
                                "test_sessions"
                            ]
                        )
                    ),
                    "session_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
        )

        print(
            f"   Fold {fold['fold']}: "
            f"Bianca teste="
            f"{','.join(bianca_test_sessions)}"
        )

    reset_outputs()

    print(
        "\n"
        + "="
        * 106
    )

    print(
        "4) TREINANDO E AVALIANDO"
    )

    print(
        "="
        * 106
    )

    all_rows = []

    for strategy in STRATEGIES:

        print(
            "\n"
            + "-"
            * 106
        )

        print(
            f"ESTRATEGIA: {strategy}"
        )

        print(
            "-"
            * 106
        )

        for fold_info in outer_folds:

            fold_number = int(
                fold_info[
                    "fold"
                ]
            )

            print(
                f"\n   Fold externo {fold_number}"
            )

            train, test = (
                build_train_test_for_strategy(
                    all_df=all_df,
                    mouse_df=mouse_df,
                    fold_info=fold_info,
                    strategy=strategy,
                )
            )

            train_balanced = (
                sample_impostors(
                    train,
                    protected_user=PROTECTED_USER,
                )
            )

            y_train = (
                train_balanced[
                    "user"
                ]
                .astype(str)
                .eq(
                    PROTECTED_USER
                )
                .astype(
                    np.int8
                )
                .to_numpy()
            )

            y_test = (
                test[
                    "user"
                ]
                .astype(str)
                .eq(
                    PROTECTED_USER
                )
                .astype(
                    np.int8
                )
                .to_numpy()
            )

            print(
                f"      treino balanceado="
                f"{len(train_balanced):,} | "
                f"genuino="
                f"{int(np.sum(y_train == 1)):,} | "
                f"impostor="
                f"{int(np.sum(y_train == 0)):,} | "
                f"teste="
                f"{len(test):,}"
                .replace(
                    ",",
                    ".",
                )
            )

            # ------------------------------------------------
            # PYCARET
            # ------------------------------------------------

            print(
                "      PyCaret selecionando "
                "LR/RF/ET no treino interno..."
            )

            try:

                py_pipeline, py_model_id = (
                    pycaret_select_on_training_only(
                        train_balanced=train_balanced,
                        features=features,
                        strategy=strategy,
                        outer_fold=fold_number,
                    )
                )

                py_candidate = (
                    f"PyCaret_{py_model_id}"
                )

                model = clone(
                    py_pipeline
                )

                start = time.perf_counter()

                model.fit(
                    train_balanced[
                        features
                    ],
                    y_train,
                )

                fit_seconds = (
                    time.perf_counter()
                    - start
                )

                scores = get_score(
                    model,
                    test[
                        features
                    ],
                )

                metrics = calculate_metrics(
                    y_test,
                    scores,
                )

                row = {
                    "strategy": strategy,
                    "candidate_family": "PYCARET_SELECTED",
                    "candidate": py_candidate,
                    "fold": fold_number,
                    "gpu_used": False,
                    "n_train": len(
                        train_balanced
                    ),
                    "n_test": len(
                        test
                    ),
                    "fit_seconds": fit_seconds,
                    **metrics,
                }

                all_rows.append(
                    row
                )

                append_df(
                    pd.DataFrame(
                        [row]
                    ),
                    FOLDS_PATH,
                )

                pred = pd.DataFrame(
                    {
                        "strategy": strategy,
                        "candidate_family": "PYCARET_SELECTED",
                        "candidate": py_candidate,
                        "fold": fold_number,
                        "session_id": test[
                            "session_id"
                        ].astype(str).to_numpy(),
                        "window_id": test[
                            "window_id"
                        ].astype(str).to_numpy(),
                        "actual_user": test[
                            "user"
                        ].astype(str).to_numpy(),
                        "y_true_genuine": y_test,
                        "score_genuine": scores,
                    }
                )

                append_df(
                    pred,
                    PREDICTIONS_PATH,
                )

                print(
                    f"         {py_candidate}: "
                    f"EER={metrics['eer']:.4f} | "
                    f"AUC={metrics['roc_auc']:.4f} | "
                    f"F1@EER="
                    f"{metrics['f1_genuine_eer']:.4f}"
                )

                del model
                del py_pipeline
                del pred
                del scores

            except Exception as exc:

                print(
                    "         AVISO PyCaret: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            gc.collect()

            # ------------------------------------------------
            # XGBOOST GPU LEVE
            # ------------------------------------------------

            print(
                "      XGBoost GPU leve..."
            )

            xgb_model = (
                build_xgb_light(
                    y_train=y_train,
                    use_gpu=use_gpu,
                )
            )

            start = time.perf_counter()

            xgb_model.fit(
                train_balanced[
                    features
                ],
                y_train,
            )

            fit_seconds = (
                time.perf_counter()
                - start
            )

            xgb_scores = get_score(
                xgb_model,
                test[
                    features
                ],
            )

            xgb_metrics = (
                calculate_metrics(
                    y_test,
                    xgb_scores,
                )
            )

            xgb_candidate = (
                "XGBoost_GPU_Light"
                if use_gpu
                else "XGBoost_CPU_Light"
            )

            xgb_row = {
                "strategy": strategy,
                "candidate_family": "XGBOOST_LIGHT",
                "candidate": xgb_candidate,
                "fold": fold_number,
                "gpu_used": bool(
                    use_gpu
                ),
                "n_train": len(
                    train_balanced
                ),
                "n_test": len(
                    test
                ),
                "fit_seconds": fit_seconds,
                **xgb_metrics,
            }

            all_rows.append(
                xgb_row
            )

            append_df(
                pd.DataFrame(
                    [
                        xgb_row
                    ]
                ),
                FOLDS_PATH,
            )

            xgb_pred = pd.DataFrame(
                {
                    "strategy": strategy,
                    "candidate_family": "XGBOOST_LIGHT",
                    "candidate": xgb_candidate,
                    "fold": fold_number,
                    "session_id": test[
                        "session_id"
                    ].astype(str).to_numpy(),
                    "window_id": test[
                        "window_id"
                    ].astype(str).to_numpy(),
                    "actual_user": test[
                        "user"
                    ].astype(str).to_numpy(),
                    "y_true_genuine": y_test,
                    "score_genuine": xgb_scores,
                }
            )

            append_df(
                xgb_pred,
                PREDICTIONS_PATH,
            )

            print(
                f"         {xgb_candidate}: "
                f"EER={xgb_metrics['eer']:.4f} | "
                f"AUC={xgb_metrics['roc_auc']:.4f} | "
                f"F1@EER="
                f"{xgb_metrics['f1_genuine_eer']:.4f}"
            )

            del xgb_model
            del xgb_pred
            del xgb_scores
            del train
            del test
            del train_balanced
            del y_train
            del y_test

            gc.collect()

            print(
                "      Memoria temporaria liberada."
            )

    # ========================================================
    # RESUMOS
    # ========================================================

    folds_df = pd.DataFrame(
        all_rows
    )

    summary_rows = []

    metrics_to_avg = [
        "eer",
        "far_at_eer",
        "frr_at_eer",
        "roc_auc",
        "f1_genuine_eer",
        "balanced_accuracy_eer",
        "accuracy_eer",
        "precision_genuine_eer",
        "recall_genuine_eer",
        "fit_seconds",
    ]

    grouped = folds_df.groupby(
        [
            "strategy",
            "candidate_family",
        ],
        sort=True,
    )

    for (
        strategy,
        candidate_family,
    ), group in grouped:

        row = {
            "strategy": strategy,
            "candidate_family": candidate_family,
            "n_folds": len(
                group
            ),
            "candidate_examples": "|".join(
                sorted(
                    group[
                        "candidate"
                    ]
                    .astype(str)
                    .unique()
                    .tolist()
                )
            ),
            "gpu_used": bool(
                group[
                    "gpu_used"
                ].any()
            ),
        }

        for metric in metrics_to_avg:

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

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
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

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    winner_rows = []

    for strategy, group in summary.groupby(
        "strategy",
        sort=True,
    ):

        ranked = group.sort_values(
            [
                "eer_mean",
                "roc_auc_mean",
                "balanced_accuracy_eer_mean",
                "f1_genuine_eer_mean",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )

        winner_rows.append(
            ranked.iloc[
                0
            ].copy()
        )

    winners = pd.DataFrame(
        winner_rows
    ).reset_index(
        drop=True
    )

    winners.to_csv(
        WINNERS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    comparison = winners[
        [
            "strategy",
            "candidate_family",
            "candidate_examples",
            "gpu_used",
            "eer_mean",
            "eer_std",
            "far_at_eer_mean",
            "frr_at_eer_mean",
            "roc_auc_mean",
            "f1_genuine_eer_mean",
            "balanced_accuracy_eer_mean",
        ]
    ].copy()

    comparison = comparison.sort_values(
        [
            "eer_mean",
            "roc_auc_mean",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    comparison.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                comparison
            )
            + 1,
        ),
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n"
        + "="
        * 106
    )

    print(
        "5) VENCEDOR POR ESTRATEGIA"
    )

    print(
        "="
        * 106
    )

    print(
        winners[
            [
                "strategy",
                "candidate_family",
                "candidate_examples",
                "eer_mean",
                "eer_std",
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
        * 106
    )

    print(
        "6) COMPARACAO FINAL"
    )

    print(
        "="
        * 106
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    best = comparison.iloc[
        0
    ]

    print(
        "\nMELHOR ESTRATEGIA:"
    )

    print(
        f"   {best['strategy']}"
    )

    print(
        f"   EER medio: "
        f"{best['eer_mean']:.4f}"
    )

    print(
        f"   AUC media: "
        f"{best['roc_auc_mean']:.4f}"
    )

    print(
        f"   F1@EER medio: "
        f"{best['f1_genuine_eer_mean']:.4f}"
    )

    manifest = {
        "projeto": "CYBELL-J",
        "fase": (
            "Experimento 11A - perfil unico "
            "versus subperfil mouse para Bianca"
        ),
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pycaret_version": pycaret.__version__,
        "protected_user": PROTECTED_USER,
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP_PCT,
        "evaluation_context": "mouse",
        "strategies": {
            "PERFIL_UNICO": (
                "Bianca mouse+trackpad como genuino no treino; "
                "impostores mouse; teste mouse."
            ),
            "SUBPERFIL_MOUSE": (
                "Bianca mouse como genuino no treino; "
                "impostores mouse; teste mouse."
            ),
        },
        "resource_profile": {
            "cpu_jobs": CPU_JOBS,
            "pycaret_use_gpu": PYCARET_USE_GPU,
            "xgboost_gpu_available": use_gpu,
            "xgboost_n_estimators": XGB_N_ESTIMATORS,
            "xgboost_max_depth": XGB_MAX_DEPTH,
            "xgboost_max_bin": XGB_MAX_BIN,
            "impostor_to_genuine_train_ratio": (
                IMPOSTOR_TO_GENUINE_TRAIN_RATIO
            ),
            "float32_features": True,
            "csv_chunk_size": CSV_CHUNK_SIZE,
            "incremental_writes": True,
            "gc_between_folds": True,
        },
        "validation": {
            "outer_folds": N_OUTER_FOLDS,
            "outer_split": (
                "session_id; teste somente mouse; "
                "cada sessao mouse aparece uma vez em teste"
            ),
            "pycaret_inner_folds": N_INNER_FOLDS,
            "pycaret_inner_split": (
                "session_id dentro do conjunto de treino externo"
            ),
            "pycaret_models": PYCARET_INCLUDE,
        },
        "best_strategy": {
            "strategy": str(
                best[
                    "strategy"
                ]
            ),
            "candidate_family": str(
                best[
                    "candidate_family"
                ]
            ),
            "eer_mean": float(
                best[
                    "eer_mean"
                ]
            ),
            "roc_auc_mean": float(
                best[
                    "roc_auc_mean"
                ]
            ),
            "f1_genuine_eer_mean": float(
                best[
                    "f1_genuine_eer_mean"
                ]
            ),
        },
        "limitations": [
            "Somente Bianca possui volume substancial nos dois dispositivos.",
            "O experimento compara desempenho no contexto mouse.",
            "Nao ha quantidade multiusuario suficiente de trackpad para validar um autenticador trackpad robusto.",
            "O perfil unico possui mais sessoes genuinas disponiveis do que o subperfil mouse; isto faz parte da comparacao pratica entre misturar e condicionar por dispositivo, mas deve ser discutido no TCC.",
            "O rótulo device_context deriva da heurística do front-end.",
        ],
        "outputs": {
            "fold_results": FOLDS_PATH.name,
            "summary": SUMMARY_PATH.name,
            "strategy_winners": WINNERS_PATH.name,
            "final_comparison": COMPARISON_PATH.name,
            "predictions_oof": PREDICTIONS_PATH.name,
            "leaderboards_pycaret": (
                LEADERBOARD_DIR.name
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
        )

    print(
        "\n"
        + "="
        * 106
    )

    print(
        "EXPERIMENTO 11A CONCLUIDO"
    )

    print(
        "="
        * 106
    )

    print(
        f"\nResultados por fold:\n"
        f"{FOLDS_PATH}"
    )

    print(
        f"\nResumo:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nVencedores por estrategia:\n"
        f"{WINNERS_PATH}"
    )

    print(
        f"\nComparacao final:\n"
        f"{COMPARISON_PATH}"
    )

    print(
        f"\nPredicoes OOF:\n"
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"\nManifesto:\n"
        f"{MANIFEST_PATH}"
    )

    print(
        "\nProximo passo, se o subperfil mouse mostrar vantagem:"
        "\n11B - agregacao temporal da comparacao "
        "perfil unico x subperfil mouse."
    )


if __name__ == "__main__":
    main()