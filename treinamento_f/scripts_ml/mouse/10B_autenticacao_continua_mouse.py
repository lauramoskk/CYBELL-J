"""
10B_autenticacao_continua_mouse.py

CYBELL-J — Experimento 10B
AUTENTICAÇÃO CONTÍNUA 1:1 POR COMPORTAMENTO DE MOUSE

Pergunta correta do produto:
    "A pessoa que está usando a sessão continua se comportando
     como a usuária que está logada, ou o comportamento ficou suspeito?"

Para cada perfil protegido:
    GENUÍNO   = janelas da própria usuária
    IMPOSTOR  = janelas das outras cinco usuárias

Exemplo:
    perfil protegido = bianca
    genuíno  = bianca
    impostor = Julia104soup, lala, laura_oliveira, lina, yass

DESENHO
-------
- Usa SOMENTE mouse de alta confiança.
- Usa a melhor configuração encontrada no 10A:
      50 eventos / 75% overlap
- Validação externa:
      3 folds por session_id
      nenhuma sessão aparece simultaneamente em treino e teste
- PyCaret é usado de forma LEVE para comparar apenas:
      Logistic Regression
      Random Forest
      Extra Trees
- O PyCaret NÃO usa GPU neste script.
- XGBoost é testado separadamente usando GPU CUDA em perfil LEVE.
- A GPU é usada APENAS pelo XGBoost e em treinos curtos.
- CPU limitada a 2 threads.
- Treino usa no máximo 2 janelas de impostor para cada janela genuína.
  Isso reduz desbalanceamento, RAM e custo computacional.
- Teste usa TODAS as janelas das sessões de teste (sem subamostragem).

MÉTRICAS PRINCIPAIS
-------------------
- EER (quanto menor, melhor)
- FAR no limiar de EER
- FRR no limiar de EER
- ROC-AUC
- F1 da classe genuína
- Balanced Accuracy
- Accuracy
- Precision / Recall

IMPORTANTE
----------
Este é o BASELINE de autenticação contínua por janela.
A agregação temporal 5/10/15/20 s deve ser feita depois (10C).

RECURSOS
--------
Perfil conservador:
- PyCaret: use_gpu=False
- CPU: n_jobs=2
- XGBoost GPU: 160 árvores, max_depth=4, max_bin=128
- features em float32
- resultados gravados progressivamente em disco
- memória liberada entre folds/perfis

SAÍDAS
------
data/mouse_authentication_1to1/
    resultados_folds.csv
    resumo_por_perfil_modelo.csv
    vencedores_por_perfil.csv
    predicoes_oof.csv
    thresholds_cv.json
    leaderboards_pycaret/
    modelos/
    manifesto_10b.json
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler

try:
    import pycaret
    try:
        from pycaret.tasks import ClassificationExperiment
    except ImportError:
        from pycaret.classification import ClassificationExperiment
except ImportError as exc:
    raise SystemExit(
        "\nERRO: PyCaret não está disponível neste ambiente.\n"
        "Ative o ambiente .venv_pycaret antes de rodar:\n"
        "    deactivate\n"
        "    .\\.venv_pycaret\\Scripts\\Activate.ps1\n"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "\nERRO: xgboost não está instalado neste ambiente.\n"
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
    / "mouse_authentication_1to1"
)

LEADERBOARD_DIR = OUTPUT_DIR / "leaderboards_pycaret"
MODELS_DIR = OUTPUT_DIR / "modelos"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FOLDS_PATH = OUTPUT_DIR / "resultados_folds.csv"
SUMMARY_PATH = OUTPUT_DIR / "resumo_por_perfil_modelo.csv"
WINNERS_PATH = OUTPUT_DIR / "vencedores_por_perfil.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predicoes_oof.csv"
THRESHOLDS_PATH = OUTPUT_DIR / "thresholds_cv.json"
MANIFEST_PATH = OUTPUT_DIR / "manifesto_10b.json"

DEVICE_CONTEXT = "mouse"

# Melhor configuração do experimento 10A
WINDOW_SIZE = 50
OVERLAP_PCT = 75

N_FOLDS = 3
RANDOM_STATE = 42

# PERFIL CONSERVADOR DE RECURSOS
CPU_JOBS = 2
PYCARET_USE_GPU = False

# Treino: no máximo 2 impostores para cada genuíno.
IMPOSTOR_TO_GENUINE_TRAIN_RATIO = 2.0

# XGBoost: uso leve da GPU
XGB_USE_GPU_IF_AVAILABLE = True
XGB_N_ESTIMATORS = 160
XGB_MAX_DEPTH = 4
XGB_MAX_BIN = 128
XGB_LEARNING_RATE = 0.06
XGB_SUBSAMPLE = 0.80
XGB_COLSAMPLE = 0.80

# PyCaret: comparação propositalmente curta
PYCARET_INCLUDE = ["lr", "rf", "et"]
PYCARET_TRAIN_SIZE = 0.95

# Evita paralelismo excessivo de bibliotecas numéricas.
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_JOBS))


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


def select_features(df: pd.DataFrame) -> list[str]:
    features = []

    for column in df.columns:
        if column in METADATA_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            features.append(column)

    if not features:
        raise RuntimeError(
            "Nenhuma feature numérica foi encontrada."
        )

    return features


def to_float32_features(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Converte somente as features para float32.
    Reduz o consumo de RAM aproximadamente pela metade em relação
    a float64 para essas colunas.
    """
    out = df.copy()

    for column in features:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        ).astype("float32")

    return out


# ============================================================
# FOLDS POR SESSÃO
# ============================================================

def stable_seed(text: str) -> int:
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        RANDOM_STATE
        + int(digest[:8], 16)
    ) % (2**32 - 1)


def build_session_folds(
    df_config: pd.DataFrame,
    n_folds: int = N_FOLDS,
) -> list[dict]:
    """
    Mesma ideia do 10A:
    distribui as sessões de cada usuária entre 3 folds.
    Cada sessão aparece em teste exatamente uma vez.
    """

    users = sorted(
        df_config["user"]
        .astype(str)
        .unique()
        .tolist()
    )

    fold_test_sessions = [
        []
        for _ in range(n_folds)
    ]

    for user in users:
        sessions = sorted(
            df_config.loc[
                df_config["user"].astype(str).eq(user),
                "session_id",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(sessions) < n_folds:
            raise RuntimeError(
                f"{user} possui apenas {len(sessions)} sessões; "
                f"são necessárias pelo menos {n_folds}."
            )

        rng = np.random.default_rng(
            stable_seed(user)
        )

        rng.shuffle(sessions)

        for idx, session in enumerate(sessions):
            fold_test_sessions[
                idx % n_folds
            ].append(session)

    all_sessions = set(
        df_config["session_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    folds = []

    for fold_idx in range(n_folds):
        test_sessions = sorted(
            set(
                fold_test_sessions[
                    fold_idx
                ]
            )
        )

        train_sessions = sorted(
            all_sessions
            - set(test_sessions)
        )

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_sessions": train_sessions,
                "test_sessions": test_sessions,
            }
        )

    # Segurança contra vazamento
    all_test = []

    for fold in folds:
        if set(
            fold["train_sessions"]
        ) & set(
            fold["test_sessions"]
        ):
            raise RuntimeError(
                "Vazamento de session_id detectado."
            )

        all_test.extend(
            fold["test_sessions"]
        )

    if len(all_test) != len(set(all_test)):
        raise RuntimeError(
            "Uma session_id apareceu em teste em mais de um fold."
        )

    if set(all_test) != all_sessions:
        raise RuntimeError(
            "Nem todas as sessões apareceram em teste."
        )

    return folds


class SessionPredefinedCV(BaseCrossValidator):
    """
    Cross-validator usado SOMENTE pelo screening interno do PyCaret.
    O resultado científico principal é recalculado depois nos folds
    externos completos por session_id.
    """

    def __init__(
        self,
        index_to_session: dict,
        fold_definitions: list[dict],
    ):
        self.index_to_session = index_to_session
        self.fold_definitions = fold_definitions

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None,
    ):
        return len(
            self.fold_definitions
        )

    def split(
        self,
        X,
        y=None,
        groups=None,
    ):
        if not hasattr(X, "index"):
            raise RuntimeError(
                "O CV por sessão exige DataFrame com índice preservado."
            )

        session_values = np.array(
            [
                self.index_to_session.get(idx)
                for idx in X.index
            ],
            dtype=object,
        )

        for fold in self.fold_definitions:
            test_sessions = set(
                fold["test_sessions"]
            )

            test_mask = np.array(
                [
                    session in test_sessions
                    for session in session_values
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
                len(train_idx) == 0
                or len(test_idx) == 0
            ):
                continue

            if y is not None:
                y_arr = np.asarray(y)

                if len(
                    np.unique(
                        y_arr[train_idx]
                    )
                ) < 2:
                    continue

                if len(
                    np.unique(
                        y_arr[test_idx]
                    )
                ) < 2:
                    continue

            yield (
                train_idx,
                test_idx,
            )


# ============================================================
# BALANCEAMENTO LEVE DO TREINO
# ============================================================

def sample_impostors_balanced(
    train: pd.DataFrame,
    protected_user: str,
    ratio: float = IMPOSTOR_TO_GENUINE_TRAIN_RATIO,
) -> pd.DataFrame:
    """
    Mantém TODOS os genuínos.
    Amostra impostores até ratio * n_genuine.
    A amostragem é distribuída entre as outras usuárias.

    Isso:
    - reduz RAM;
    - reduz CPU/GPU;
    - evita que o modelo veja 10x mais impostores que genuínos.
    """

    genuine = train[
        train["user"]
        .astype(str)
        .eq(protected_user)
    ].copy()

    impostors = train[
        ~train["user"]
        .astype(str)
        .eq(protected_user)
    ].copy()

    if genuine.empty:
        raise RuntimeError(
            f"Sem genuínos no treino para {protected_user}."
        )

    target_impostors = int(
        round(
            len(genuine)
            * ratio
        )
    )

    impostor_users = sorted(
        impostors["user"]
        .astype(str)
        .unique()
        .tolist()
    )

    if not impostor_users:
        raise RuntimeError(
            "Sem impostores disponíveis."
        )

    per_user_target = max(
        1,
        target_impostors
        // len(impostor_users),
    )

    sampled_parts = []

    remaining = target_impostors

    for idx, impostor_user in enumerate(
        impostor_users
    ):
        pool = impostors[
            impostors["user"]
            .astype(str)
            .eq(impostor_user)
        ]

        users_left = (
            len(impostor_users)
            - idx
        )

        desired = min(
            len(pool),
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
                f"{protected_user}|{impostor_user}|{len(train)}"
            ),
        )

        sampled_parts.append(
            sampled
        )

        remaining -= len(sampled)

    sampled_impostors = pd.concat(
        sampled_parts,
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
            f"shuffle|{protected_user}|{len(train)}"
        ),
    )

    return balanced


# ============================================================
# GPU XGBOOST — PERFIL LEVE
# ============================================================

def gpu_xgboost_available() -> bool:
    if not XGB_USE_GPU_IF_AVAILABLE:
        return False

    try:
        X = np.random.default_rng(
            RANDOM_STATE
        ).random(
            (256, 8),
            dtype=np.float32,
        )

        y = np.random.default_rng(
            RANDOM_STATE + 1
        ).integers(
            0,
            2,
            size=256,
        )

        model = XGBClassifier(
            n_estimators=2,
            max_depth=2,
            tree_method="hist",
            device="cuda",
            max_bin=64,
            n_jobs=1,
            verbosity=0,
        )

        model.fit(
            X,
            y,
        )

        del model
        del X
        del y
        gc.collect()

        return True

    except Exception as exc:
        print(
            "\nAVISO: CUDA do XGBoost não ficou disponível "
            f"neste ambiente: {type(exc).__name__}: {exc}"
        )

        return False


def build_light_xgboost(
    y_train: np.ndarray,
    use_gpu: bool,
) -> Pipeline:
    """
    XGBoost deliberadamente moderado.
    GPU somente em curtos bursts.
    """

    positives = int(
        np.sum(
            y_train == 1
        )
    )

    negatives = int(
        np.sum(
            y_train == 0
        )
    )

    scale_pos_weight = (
        negatives / positives
        if positives > 0
        else 1.0
    )

    params = dict(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        min_child_weight=2,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        max_bin=XGB_MAX_BIN,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=CPU_JOBS,
        verbosity=0,
    )

    if use_gpu:
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"

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
                    **params
                ),
            ),
        ]
    )


# ============================================================
# MÉTRICAS DE AUTENTICAÇÃO
# ============================================================

def compute_eer(
    y_true: np.ndarray,
    scores_genuine: np.ndarray,
) -> dict:
    """
    y=1 = genuíno
    y=0 = impostor

    FAR = impostor aceito como genuíno = FPR
    FRR = genuíno rejeitado = FNR
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores_genuine,
        dtype=float,
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
                fpr - fnr
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
        (far + frr)
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


def calculate_auth_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
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
        np.unique(
            y_true
        )
    ) < 2:
        raise RuntimeError(
            "Teste binário sem as duas classes."
        )

    eer_info = compute_eer(
        y_true,
        scores,
    )

    pred_05 = (
        scores >= 0.5
    ).astype(int)

    pred_eer = (
        scores
        >= eer_info[
            "eer_threshold"
        ]
    ).astype(int)

    return {
        "n_test": int(
            len(y_true)
        ),
        "n_genuine": int(
            np.sum(
                y_true == 1
            )
        ),
        "n_impostor": int(
            np.sum(
                y_true == 0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        **eer_info,
        "f1_genuine_05": float(
            f1_score(
                y_true,
                pred_05,
                pos_label=1,
                zero_division=0,
            )
        ),
        "balanced_accuracy_05": float(
            balanced_accuracy_score(
                y_true,
                pred_05,
            )
        ),
        "accuracy_05": float(
            accuracy_score(
                y_true,
                pred_05,
            )
        ),
        "precision_genuine_05": float(
            precision_score(
                y_true,
                pred_05,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_genuine_05": float(
            recall_score(
                y_true,
                pred_05,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_genuine_eer": float(
            f1_score(
                y_true,
                pred_eer,
                pos_label=1,
                zero_division=0,
            )
        ),
        "balanced_accuracy_eer": float(
            balanced_accuracy_score(
                y_true,
                pred_eer,
            )
        ),
        "accuracy_eer": float(
            accuracy_score(
                y_true,
                pred_eer,
            )
        ),
    }


def get_score_genuine(
    model,
    X: pd.DataFrame,
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
                [0, 1],
            )
        )

        if 1 in classes:
            idx = classes.index(1)
        else:
            idx = 1

        return proba[
            :,
            idx
        ].astype(float)

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


# ============================================================
# CSV INCREMENTAL
# ============================================================

def reset_output_csvs():
    for path in [
        FOLDS_PATH,
        PREDICTIONS_PATH,
    ]:
        if path.exists():
            path.unlink()


def append_dataframe(
    df: pd.DataFrame,
    path: Path,
):
    write_header = not path.exists()

    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# PYCARET SCREENING LEVE
# ============================================================

def run_pycaret_screening(
    profile_df: pd.DataFrame,
    features: list[str],
    protected_user: str,
    folds: list[dict],
) -> tuple[
    BaseEstimator,
    str,
    pd.DataFrame,
]:
    """
    PyCaret compara SOMENTE lr/rf/et, CPU, 2 threads.
    Retorna o melhor pipeline por AUC.
    """

    screening_df = sample_impostors_balanced(
        profile_df,
        protected_user=protected_user,
    ).copy()

    screening_df["target_bin"] = (
        screening_df["user"]
        .astype(str)
        .eq(
            protected_user
        )
        .astype(int)
    )

    index_to_session = (
        screening_df[
            "session_id"
        ]
        .astype(str)
        .to_dict()
    )

    session_cv = SessionPredefinedCV(
        index_to_session=index_to_session,
        fold_definitions=folds,
    )

    py_data = screening_df[
        features
        + ["target_bin"]
    ].copy()

    exp = ClassificationExperiment(
        target="target_bin",
        session_id=RANDOM_STATE,
        train_size=PYCARET_TRAIN_SIZE,
        fold=N_FOLDS,
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
        fold=session_cv,
        include=PYCARET_INCLUDE,
        sort="AUC",
        n_select=1,
        turbo=True,
        errors="ignore",
        round=4,
        verbose=False,
    )

    leaderboard = result.leaderboard.copy()

    leaderboard_path = (
        LEADERBOARD_DIR
        / f"{protected_user}_leaderboard.csv"
    )

    leaderboard.to_csv(
        leaderboard_path,
        index=False,
        encoding="utf-8-sig",
    )

    if not result.models:
        raise RuntimeError(
            f"PyCaret não retornou modelo para {protected_user}."
        )

    pipeline = result.models[0]

    model_id = (
        result.ranked_ids[0]
        if result.ranked_ids
        else "pycaret_best"
    )

    # O pipeline será clonado nos folds externos.
    del exp
    del result
    del py_data
    del screening_df

    gc.collect()

    return (
        pipeline,
        str(model_id),
        leaderboard,
    )


# ============================================================
# AVALIAÇÃO EXTERNA POR PERFIL
# ============================================================

def evaluate_candidate(
    candidate_name: str,
    pipeline_template,
    df_config: pd.DataFrame,
    protected_user: str,
    features: list[str],
    folds: list[dict],
    is_xgboost_gpu: bool,
) -> list[dict]:

    rows = []

    for fold_info in folds:
        fold_number = int(
            fold_info["fold"]
        )

        train = df_config[
            df_config["session_id"]
            .astype(str)
            .isin(
                fold_info[
                    "train_sessions"
                ]
            )
        ].copy()

        test = df_config[
            df_config["session_id"]
            .astype(str)
            .isin(
                fold_info[
                    "test_sessions"
                ]
            )
        ].copy()

        train_balanced = (
            sample_impostors_balanced(
                train,
                protected_user=protected_user,
            )
        )

        y_train = (
            train_balanced["user"]
            .astype(str)
            .eq(
                protected_user
            )
            .astype(int)
            .to_numpy()
        )

        y_test = (
            test["user"]
            .astype(str)
            .eq(
                protected_user
            )
            .astype(int)
            .to_numpy()
        )

        X_train = train_balanced[
            features
        ]

        X_test = test[
            features
        ]

        if is_xgboost_gpu:
            model = build_light_xgboost(
                y_train=y_train,
                use_gpu=True,
            )
        else:
            model = clone(
                pipeline_template
            )

        start_fit = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        fit_seconds = (
            time.perf_counter()
            - start_fit
        )

        start_pred = time.perf_counter()

        scores = get_score_genuine(
            model,
            X_test,
        )

        pred_seconds = (
            time.perf_counter()
            - start_pred
        )

        metrics = calculate_auth_metrics(
            y_test,
            scores,
        )

        row = {
            "protected_user": protected_user,
            "candidate": candidate_name,
            "fold": fold_number,
            "gpu_used": bool(
                is_xgboost_gpu
            ),
            "n_features": int(
                len(features)
            ),
            "n_train_after_sampling": int(
                len(train_balanced)
            ),
            "n_train_genuine": int(
                np.sum(
                    y_train == 1
                )
            ),
            "n_train_impostor": int(
                np.sum(
                    y_train == 0
                )
            ),
            "fit_seconds": float(
                fit_seconds
            ),
            "predict_ms_per_window": float(
                pred_seconds
                / max(
                    len(test),
                    1,
                )
                * 1000.0
            ),
            "train_sessions": "|".join(
                fold_info[
                    "train_sessions"
                ]
            ),
            "test_sessions": "|".join(
                fold_info[
                    "test_sessions"
                ]
            ),
            **metrics,
        }

        rows.append(
            row
        )

        append_dataframe(
            pd.DataFrame(
                [row]
            ),
            FOLDS_PATH,
        )

        prediction_df = pd.DataFrame(
            {
                "protected_user": protected_user,
                "candidate": candidate_name,
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
                "eer_threshold_fold": metrics[
                    "eer_threshold"
                ],
            }
        )

        append_dataframe(
            prediction_df,
            PREDICTIONS_PATH,
        )

        print(
            f"      Fold {fold_number} | "
            f"EER={metrics['eer']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f} | "
            f"F1@EER={metrics['f1_genuine_eer']:.4f} | "
            f"BalAcc@EER={metrics['balanced_accuracy_eer']:.4f}"
        )

        del model
        del train
        del test
        del train_balanced
        del X_train
        del X_test
        del y_train
        del y_test
        del scores
        del prediction_df

        gc.collect()

    return rows


# ============================================================
# RESUMOS
# ============================================================

def summarize_results(
    folds_df: pd.DataFrame,
) -> pd.DataFrame:

    metric_columns = [
        "eer",
        "far_at_eer",
        "frr_at_eer",
        "roc_auc",
        "f1_genuine_05",
        "balanced_accuracy_05",
        "accuracy_05",
        "precision_genuine_05",
        "recall_genuine_05",
        "f1_genuine_eer",
        "balanced_accuracy_eer",
        "accuracy_eer",
        "eer_threshold",
        "fit_seconds",
        "predict_ms_per_window",
    ]

    rows = []

    grouped = folds_df.groupby(
        [
            "protected_user",
            "candidate",
            "gpu_used",
        ],
        sort=True,
        dropna=False,
    )

    for keys, group in grouped:
        (
            protected_user,
            candidate,
            gpu_used,
        ) = keys

        row = {
            "protected_user": protected_user,
            "candidate": candidate,
            "gpu_used": bool(
                gpu_used
            ),
            "n_folds": int(
                len(group)
            ),
        }

        for metric in metric_columns:
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

    summary = summary.sort_values(
        [
            "protected_user",
            "eer_mean",
            "roc_auc_mean",
            "balanced_accuracy_eer_mean",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return summary


def choose_winners(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    winners = []

    for protected_user, group in summary.groupby(
        "protected_user",
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

        best = ranked.iloc[0].copy()

        winners.append(
            best
        )

    winners_df = pd.DataFrame(
        winners
    ).reset_index(
        drop=True
    )

    return winners_df


# ============================================================
# MODELO FINAL POR PERFIL
# ============================================================

def train_final_profile_model(
    df_config: pd.DataFrame,
    protected_user: str,
    winner_row: pd.Series,
    pycaret_pipeline,
    pycaret_candidate_name: str,
    features: list[str],
    gpu_available: bool,
):
    """
    Treina o candidato vencedor em um conjunto de treino balanceado
    construído a partir de TODAS as sessões.

    O threshold salvo é a MEDIANA dos thresholds de EER obtidos nos
    folds externos. É um threshold de pesquisa/candidato; para produção,
    deve ser recalibrado em conjunto de validação separado.
    """

    train_balanced = sample_impostors_balanced(
        df_config,
        protected_user=protected_user,
    )

    y = (
        train_balanced["user"]
        .astype(str)
        .eq(
            protected_user
        )
        .astype(int)
        .to_numpy()
    )

    X = train_balanced[
        features
    ]

    winner_name = str(
        winner_row[
            "candidate"
        ]
    )

    if winner_name == "XGBoost_GPU_Light":
        model = build_light_xgboost(
            y_train=y,
            use_gpu=gpu_available,
        )
    elif winner_name == pycaret_candidate_name:
        model = clone(
            pycaret_pipeline
        )
    else:
        raise RuntimeError(
            f"Vencedor desconhecido: {winner_name}"
        )

    model.fit(
        X,
        y,
    )

    bundle = {
        "model": model,
        "protected_user": protected_user,
        "features": features,
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP_PCT,
        "device_context": DEVICE_CONTEXT,
        "candidate": winner_name,
        "threshold_cv_median_candidate": float(
            winner_row[
                "eer_threshold_mean"
            ]
        ),
        "threshold_note": (
            "Threshold de pesquisa derivado dos folds; "
            "recalibrar antes de produção."
        ),
    }

    model_path = (
        MODELS_DIR
        / f"{protected_user}_authenticator.joblib"
    )

    joblib.dump(
        bundle,
        model_path,
    )

    del model
    del bundle
    del train_balanced
    del X
    del y

    gc.collect()

    return model_path


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "="
        * 100
    )

    print(
        "CYBELL-J — EXPERIMENTO 10B — "
        "AUTENTICACAO CONTINUA 1:1"
    )

    print(
        "="
        * 100
    )

    print(
        f"\nPyCaret: {pycaret.__version__}"
        f"\nCPU limitada a: {CPU_JOBS} threads"
        f"\nPyCaret usa GPU? {PYCARET_USE_GPU}"
        f"\nXGBoost GPU: perfil LEVE"
        f"\nTreino impostor:genuino <= "
        f"{IMPOSTOR_TO_GENUINE_TRAIN_RATIO}:1"
    )

    if not str(
        pycaret.__version__
    ).startswith(
        "4."
    ):
        print(
            "\nAVISO: este script foi preparado "
            "para PyCaret 4.x."
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado:\n"
            f"{DATASET_PATH}"
        )

    print(
        "\n1) Testando CUDA do XGBoost "
        "(teste minúsculo)..."
    )

    gpu_available = gpu_xgboost_available()

    if gpu_available:
        print(
            "   GPU disponível. "
            "Será usada SOMENTE pelo XGBoost "
            "em perfil leve."
        )
    else:
        print(
            "   GPU não disponível neste ambiente. "
            "XGBoost usará CPU limitada."
        )

    print(
        "\n2) Carregando dataset..."
    )

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
        "device_context",
    }

    missing = sorted(
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Colunas ausentes: {missing}"
        )

    df = df[
        (
            df["device_context"]
            .astype(str)
            .eq(
                DEVICE_CONTEXT
            )
        )
        & (
            df["window_size"]
            == WINDOW_SIZE
        )
        & (
            df["requested_overlap_pct"]
            == OVERLAP_PCT
        )
    ].copy()

    if df.empty:
        raise RuntimeError(
            "Nenhuma janela encontrada "
            f"para mouse / {WINDOW_SIZE} / {OVERLAP_PCT}%."
        )

    users = sorted(
        df["user"]
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"   Janelas: "
        f"{len(df):,}"
        .replace(",", ".")
    )

    print(
        f"   Usuárias: "
        f"{len(users)}"
    )

    print(
        "   "
        + ", ".join(
            users
        )
    )

    print(
        "\n3) Janelas por usuária:"
    )

    print(
        df.groupby(
            "user"
        )
        .size()
        .to_string()
    )

    features = select_features(
        df
    )

    print(
        f"\n4) Features: "
        f"{len(features)}"
    )

    print(
        "   Convertendo features para float32 "
        "para reduzir RAM..."
    )

    df = to_float32_features(
        df,
        features,
    )

    folds = build_session_folds(
        df,
        n_folds=N_FOLDS,
    )

    print(
        "\n5) Folds por sessão:"
    )

    for fold in folds:
        print(
            f"   Fold {fold['fold']}: "
            f"{len(fold['test_sessions'])} sessões em teste"
        )

    reset_output_csvs()

    all_summary_rows = []
    threshold_payload = {}
    profile_registry = {}

    print(
        "\n"
        + "="
        * 100
    )

    print(
        "6) TREINANDO AUTENTICADORES 1:1"
    )

    print(
        "="
        * 100
    )

    for user_index, protected_user in enumerate(
        users,
        start=1,
    ):

        print(
            "\n"
            + "-"
            * 100
        )

        print(
            f"[{user_index}/{len(users)}] "
            f"PERFIL PROTEGIDO: {protected_user}"
        )

        print(
            "-"
            * 100
        )

        print(
            "\n   A) PyCaret leve "
            "(LR / RF / ExtraTrees; CPU)"
        )

        try:
            (
                pycaret_pipeline,
                pycaret_model_id,
                leaderboard,
            ) = run_pycaret_screening(
                profile_df=df,
                features=features,
                protected_user=protected_user,
                folds=folds,
            )

            pycaret_candidate_name = (
                f"PyCaret_{pycaret_model_id}"
            )

            print(
                f"      Melhor PyCaret: "
                f"{pycaret_candidate_name}"
            )

        except Exception as exc:
            print(
                "      AVISO: screening PyCaret falhou "
                f"para {protected_user}: "
                f"{type(exc).__name__}: {exc}"
            )

            # fallback leve: Logistic Regression
            pycaret_candidate_name = (
                "Fallback_LogisticRegression"
            )

            pycaret_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=600,
                            solver="liblinear",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )

        print(
            "\n   B) Avaliação externa "
            f"{pycaret_candidate_name}"
        )

        pycaret_rows = evaluate_candidate(
            candidate_name=pycaret_candidate_name,
            pipeline_template=pycaret_pipeline,
            df_config=df,
            protected_user=protected_user,
            features=features,
            folds=folds,
            is_xgboost_gpu=False,
        )

        print(
            "\n   C) Avaliação externa "
            "XGBoost GPU leve"
        )

        xgb_rows = []

        if gpu_available:
            xgb_rows = evaluate_candidate(
                candidate_name="XGBoost_GPU_Light",
                pipeline_template=None,
                df_config=df,
                protected_user=protected_user,
                features=features,
                folds=folds,
                is_xgboost_gpu=True,
            )
        else:
            # CPU fallback com mesma configuração leve
            for fold_info in folds:
                fold_number = int(
                    fold_info["fold"]
                )

                train = df[
                    df["session_id"]
                    .astype(str)
                    .isin(
                        fold_info[
                            "train_sessions"
                        ]
                    )
                ].copy()

                test = df[
                    df["session_id"]
                    .astype(str)
                    .isin(
                        fold_info[
                            "test_sessions"
                        ]
                    )
                ].copy()

                train_balanced = (
                    sample_impostors_balanced(
                        train,
                        protected_user,
                    )
                )

                y_train = (
                    train_balanced["user"]
                    .astype(str)
                    .eq(
                        protected_user
                    )
                    .astype(int)
                    .to_numpy()
                )

                y_test = (
                    test["user"]
                    .astype(str)
                    .eq(
                        protected_user
                    )
                    .astype(int)
                    .to_numpy()
                )

                model = build_light_xgboost(
                    y_train=y_train,
                    use_gpu=False,
                )

                model.fit(
                    train_balanced[
                        features
                    ],
                    y_train,
                )

                scores = get_score_genuine(
                    model,
                    test[
                        features
                    ],
                )

                metrics = calculate_auth_metrics(
                    y_test,
                    scores,
                )

                row = {
                    "protected_user": protected_user,
                    "candidate": "XGBoost_CPU_Light",
                    "fold": fold_number,
                    "gpu_used": False,
                    "n_features": len(
                        features
                    ),
                    "n_train_after_sampling": len(
                        train_balanced
                    ),
                    "n_train_genuine": int(
                        np.sum(
                            y_train == 1
                        )
                    ),
                    "n_train_impostor": int(
                        np.sum(
                            y_train == 0
                        )
                    ),
                    "fit_seconds": float(
                        "nan"
                    ),
                    "predict_ms_per_window": float(
                        "nan"
                    ),
                    "train_sessions": "|".join(
                        fold_info[
                            "train_sessions"
                        ]
                    ),
                    "test_sessions": "|".join(
                        fold_info[
                            "test_sessions"
                        ]
                    ),
                    **metrics,
                }

                xgb_rows.append(
                    row
                )

                append_dataframe(
                    pd.DataFrame(
                        [row]
                    ),
                    FOLDS_PATH,
                )

                prediction_df = pd.DataFrame(
                    {
                        "protected_user": protected_user,
                        "candidate": "XGBoost_CPU_Light",
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
                        "eer_threshold_fold": metrics[
                            "eer_threshold"
                        ],
                    }
                )

                append_dataframe(
                    prediction_df,
                    PREDICTIONS_PATH,
                )

                del model
                del train
                del test
                del train_balanced
                del prediction_df

                gc.collect()

        profile_fold_df = pd.DataFrame(
            pycaret_rows
            + xgb_rows
        )

        profile_summary = summarize_results(
            profile_fold_df
        )

        profile_winner = choose_winners(
            profile_summary
        ).iloc[0]

        print(
            "\n   MELHOR PARA ESTE PERFIL:"
        )

        print(
            f"      Modelo: "
            f"{profile_winner['candidate']}"
        )

        print(
            f"      EER medio: "
            f"{profile_winner['eer_mean']:.4f}"
        )

        print(
            f"      FAR@EER: "
            f"{profile_winner['far_at_eer_mean']:.4f}"
        )

        print(
            f"      FRR@EER: "
            f"{profile_winner['frr_at_eer_mean']:.4f}"
        )

        print(
            f"      AUC: "
            f"{profile_winner['roc_auc_mean']:.4f}"
        )

        print(
            f"      F1@EER: "
            f"{profile_winner['f1_genuine_eer_mean']:.4f}"
        )

        threshold_payload[
            protected_user
        ] = {
            "candidate": str(
                profile_winner[
                    "candidate"
                ]
            ),
            "eer_mean": float(
                profile_winner[
                    "eer_mean"
                ]
            ),
            "threshold_cv_mean": float(
                profile_winner[
                    "eer_threshold_mean"
                ]
            ),
            "threshold_note": (
                "Threshold de pesquisa; "
                "recalibrar antes de produção."
            ),
        }

        # Salva resumo parcial já
        all_summary_rows.append(
            profile_summary
        )

        partial_summary = pd.concat(
            all_summary_rows,
            ignore_index=True,
        )

        partial_summary.to_csv(
            SUMMARY_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        # Treina e salva modelo final do perfil
        model_path = train_final_profile_model(
            df_config=df,
            protected_user=protected_user,
            winner_row=profile_winner,
            pycaret_pipeline=pycaret_pipeline,
            pycaret_candidate_name=pycaret_candidate_name,
            features=features,
            gpu_available=gpu_available,
        )

        profile_registry[
            protected_user
        ] = {
            "model_path": str(
                model_path.name
            ),
            "winner": str(
                profile_winner[
                    "candidate"
                ]
            ),
        }

        del pycaret_pipeline
        del profile_fold_df
        del profile_summary
        del profile_winner
        del pycaret_rows
        del xgb_rows

        gc.collect()

        print(
            "   Memória temporária liberada."
        )

    # ========================================================
    # RESUMOS FINAIS
    # ========================================================

    folds_df = pd.read_csv(
        FOLDS_PATH,
        low_memory=False,
    )

    summary = summarize_results(
        folds_df
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    winners = choose_winners(
        summary
    )

    winners.to_csv(
        WINNERS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    with THRESHOLDS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            threshold_payload,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(
        "\n"
        + "="
        * 100
    )

    print(
        "7) VENCEDORES POR PERFIL"
    )

    print(
        "="
        * 100
    )

    print(
        winners[
            [
                "protected_user",
                "candidate",
                "gpu_used",
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

    overall = {
        "eer_mean_profiles": float(
            winners[
                "eer_mean"
            ].mean()
        ),
        "eer_median_profiles": float(
            winners[
                "eer_mean"
            ].median()
        ),
        "auc_mean_profiles": float(
            winners[
                "roc_auc_mean"
            ].mean()
        ),
        "f1_eer_mean_profiles": float(
            winners[
                "f1_genuine_eer_mean"
            ].mean()
        ),
    }

    print(
        "\n"
        + "="
        * 100
    )

    print(
        "8) RESULTADO GLOBAL 10B"
    )

    print(
        "="
        * 100
    )

    print(
        f"\nEER medio entre perfis: "
        f"{overall['eer_mean_profiles']:.4f}"
    )

    print(
        f"EER mediano entre perfis: "
        f"{overall['eer_median_profiles']:.4f}"
    )

    print(
        f"AUC media entre perfis: "
        f"{overall['auc_mean_profiles']:.4f}"
    )

    print(
        f"F1@EER medio entre perfis: "
        f"{overall['f1_eer_mean_profiles']:.4f}"
    )

    manifest = {
        "projeto": "CYBELL-J",
        "fase": (
            "Experimento 10B - autenticacao continua "
            "1:1 por comportamento de mouse"
        ),
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pycaret_version": pycaret.__version__,
        "dataset": DATASET_PATH.name,
        "device_context": DEVICE_CONTEXT,
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP_PCT,
        "users": users,
        "features_count": len(
            features
        ),
        "resource_profile": {
            "cpu_jobs": CPU_JOBS,
            "pycaret_use_gpu": PYCARET_USE_GPU,
            "xgboost_gpu_available": gpu_available,
            "xgboost_gpu_profile": {
                "n_estimators": XGB_N_ESTIMATORS,
                "max_depth": XGB_MAX_DEPTH,
                "max_bin": XGB_MAX_BIN,
                "subsample": XGB_SUBSAMPLE,
                "colsample_bytree": XGB_COLSAMPLE,
            },
            "impostor_to_genuine_train_ratio": (
                IMPOSTOR_TO_GENUINE_TRAIN_RATIO
            ),
            "float32_features": True,
            "incremental_csv_writes": True,
        },
        "pycaret_candidates": PYCARET_INCLUDE,
        "xgboost_candidate": (
            "XGBoost_GPU_Light"
            if gpu_available
            else "XGBoost_CPU_Light"
        ),
        "metrics_primary": [
            "EER",
            "FAR@EER",
            "FRR@EER",
            "ROC-AUC",
            "F1 genuine",
            "Balanced Accuracy",
        ],
        "overall": overall,
        "profile_registry": profile_registry,
        "methodological_notes": [
            "O problema e verificacao 1:1, nao identificacao multiclasses.",
            "Classe positiva = comportamento genuino da identidade logada.",
            "Classe negativa = comportamento das outras cinco usuarias.",
            "Treino usa subamostragem de impostores ate 2:1 para reduzir desbalanceamento e custo computacional.",
            "Teste usa todas as janelas das sessoes de teste.",
            "Nenhuma session_id aparece simultaneamente em treino e teste.",
            "PyCaret compara apenas LR, Random Forest e Extra Trees em CPU limitada.",
            "XGBoost e avaliado separadamente com GPU em configuracao leve quando CUDA esta disponivel.",
            "O threshold de EER calculado nos folds e de pesquisa; para producao deve ser calibrado em conjunto de validacao independente.",
            "A agregacao temporal 5/10/15/20 s nao faz parte deste baseline e deve ser avaliada posteriormente.",
        ],
        "outputs": {
            "fold_results": FOLDS_PATH.name,
            "summary": SUMMARY_PATH.name,
            "winners": WINNERS_PATH.name,
            "predictions_oof": PREDICTIONS_PATH.name,
            "thresholds": THRESHOLDS_PATH.name,
            "leaderboards_dir": LEADERBOARD_DIR.name,
            "models_dir": MODELS_DIR.name,
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
        * 100
    )

    print(
        "EXPERIMENTO 10B CONCLUIDO"
    )

    print(
        "="
        * 100
    )

    print(
        f"\nResultados por fold:\n"
        f"{FOLDS_PATH}"
    )

    print(
        f"\nResumo por perfil/modelo:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nVencedores por perfil:\n"
        f"{WINNERS_PATH}"
    )

    print(
        f"\nPredicoes OOF:\n"
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"\nThresholds:\n"
        f"{THRESHOLDS_PATH}"
    )

    print(
        f"\nModelos finais:\n"
        f"{MODELS_DIR}"
    )

    print(
        "\nProximo passo: 10C - agregacao temporal "
        "dos scores de autenticacao em 5/10/15/20 s."
    )


if __name__ == "__main__":
    main()