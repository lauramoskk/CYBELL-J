"""
07_hiperparametrizacao_teclado.py

CYBELL-J — Hiperparametrização do modelo GLOBAL de teclado

Objetivo:
- usar somente teclado;
- usar a configuração vencedora do Experimento 2:
  janela = 50 eventos, overlap = 75%;
- usar exatamente as 59 features do manifesto do teclado;
- manter o mesmo pré-processamento do mouse:
  SimpleImputer(strategy="median"), sem StandardScaler;
- realizar busca formal de hiperparâmetros sem vazamento entre sessões.

Estratégia:
- validação externa: 3 folds por session_id;
- busca interna: 2 folds por session_id somente dentro do treino externo;
- seleção pelo Macro F1;
- algoritmos: Random Forest e XGBoost.

Saídas:
data/teclado_hiperparametrizacao/
    candidatos_inner.csv
    resultados_outer_folds.csv
    resumo_hiperparametros.csv
    melhores_hiperparametros.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "teclado_dataset"
    / "dataset_teclado.csv"
)

DATASET_MANIFEST_PATH = (
    BASE_DIR
    / "data"
    / "teclado_dataset"
    / "dataset_teclado_manifest.json"
)

OUT_DIR = (
    BASE_DIR
    / "data"
    / "teclado_hiperparametrizacao"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

INNER_PATH = OUT_DIR / "candidatos_inner.csv"
OUTER_PATH = OUT_DIR / "resultados_outer_folds.csv"
SUMMARY_PATH = OUT_DIR / "resumo_hiperparametros.csv"
BEST_PATH = OUT_DIR / "melhores_hiperparametros.json"

TARGET_COLUMN = "user"

WINDOW_SIZE = 50
OVERLAP_PCT = 75

OUTER_FOLDS = 3
INNER_FOLDS = 2
RANDOM_STATE = 42


# Inclui os hiperparâmetros originais usados no Experimento 2
# e variações controladas para a busca.
RF_CANDIDATES = [
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
    },
    {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.7,
    },
]


XGB_CANDIDATES = [
    {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },
    {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.10,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
    },
    {
        "n_estimators": 450,
        "learning_rate": 0.03,
        "max_depth": 8,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 4,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
    },
    {
        "n_estimators": 300,
        "learning_rate": 0.07,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
    },
]


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        RANDOM_STATE
        + int(digest[:8], 16)
    ) % (2**32 - 1)


def load_features() -> list[str]:
    if not DATASET_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifesto do teclado não encontrado:\n"
            f"{DATASET_MANIFEST_PATH}"
        )

    with DATASET_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    features = manifest.get(
        "feature_columns",
        [],
    )

    if not features:
        raise RuntimeError(
            "feature_columns não encontrada no manifesto."
        )

    return list(features)


def build_session_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed_prefix: str,
) -> list[dict]:

    users = sorted(
        df[TARGET_COLUMN]
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
            df.loc[
                df[TARGET_COLUMN]
                .astype(str)
                .eq(user),
                "session_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(sessions) < n_folds:
            raise RuntimeError(
                f"{user}: apenas {len(sessions)} sessões; "
                f"são necessárias pelo menos {n_folds}."
            )

        rng = np.random.default_rng(
            stable_seed(
                f"{seed_prefix}|{user}"
            )
        )
        rng.shuffle(sessions)

        for idx, session in enumerate(sessions):
            fold_test_sessions[
                idx % n_folds
            ].append(session)

    all_sessions = set(
        df["session_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    folds = []

    for fold_idx in range(n_folds):
        test_sessions = set(
            fold_test_sessions[
                fold_idx
            ]
        )

        train_sessions = (
            all_sessions
            - test_sessions
        )

        if train_sessions & test_sessions:
            raise RuntimeError(
                "Vazamento de session_id detectado."
            )

        train = df[
            df["session_id"]
            .astype(str)
            .isin(train_sessions)
        ]

        test = df[
            df["session_id"]
            .astype(str)
            .isin(test_sessions)
        ]

        needed = set(users)

        if set(
            train[TARGET_COLUMN]
            .astype(str)
            .unique()
        ) != needed:
            raise RuntimeError(
                f"Fold {fold_idx + 1}: "
                f"treino sem todas as usuárias."
            )

        if set(
            test[TARGET_COLUMN]
            .astype(str)
            .unique()
        ) != needed:
            raise RuntimeError(
                f"Fold {fold_idx + 1}: "
                f"teste sem todas as usuárias."
            )

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_sessions": sorted(
                    train_sessions
                ),
                "test_sessions": sorted(
                    test_sessions
                ),
            }
        )

    return folds


def build_model(
    algorithm: str,
    params: dict,
    n_classes: int,
    seed: int,
) -> Pipeline:

    if algorithm == "RandomForest":
        model = RandomForestClassifier(
            **params,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )

    elif algorithm == "XGBoost":
        model = XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )

    else:
        raise RuntimeError(
            f"Algoritmo desconhecido: {algorithm}"
        )

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
                model,
            ),
        ]
    )


def fit_model(
    model: Pipeline,
    algorithm: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> None:

    if algorithm == "XGBoost":
        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        model.fit(
            X_train,
            y_train,
            model__sample_weight=weights,
        )

    else:
        model.fit(
            X_train,
            y_train,
        )


def evaluate_candidate_inner(
    outer_train: pd.DataFrame,
    features: list[str],
    encoder: LabelEncoder,
    algorithm: str,
    params: dict,
    outer_fold: int,
    candidate_id: int,
) -> tuple[float, list[float]]:

    inner_folds = build_session_folds(
        outer_train,
        n_folds=INNER_FOLDS,
        seed_prefix=(
            f"inner|outer{outer_fold}"
        ),
    )

    scores = []

    for inner_info in inner_folds:
        inner_fold = int(
            inner_info["fold"]
        )

        train = outer_train[
            outer_train["session_id"]
            .astype(str)
            .isin(
                inner_info[
                    "train_sessions"
                ]
            )
        ]

        valid = outer_train[
            outer_train["session_id"]
            .astype(str)
            .isin(
                inner_info[
                    "test_sessions"
                ]
            )
        ]

        X_train = train[
            features
        ].copy()

        X_valid = valid[
            features
        ].copy()

        y_train = encoder.transform(
            train[
                TARGET_COLUMN
            ].astype(str)
        )

        y_valid = encoder.transform(
            valid[
                TARGET_COLUMN
            ].astype(str)
        )

        model = build_model(
            algorithm=algorithm,
            params=params,
            n_classes=len(
                encoder.classes_
            ),
            seed=stable_seed(
                f"{algorithm}|outer{outer_fold}|"
                f"candidate{candidate_id}|"
                f"inner{inner_fold}"
            ),
        )

        fit_model(
            model,
            algorithm,
            X_train,
            y_train,
        )

        pred = model.predict(
            X_valid
        )

        score = f1_score(
            y_valid,
            pred,
            average="macro",
            zero_division=0,
        )

        scores.append(
            float(score)
        )

    return (
        float(
            np.mean(scores)
        ),
        scores,
    )


def evaluate_outer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    encoder: LabelEncoder,
    algorithm: str,
    params: dict,
    outer_fold: int,
) -> dict:

    X_train = train[
        features
    ].copy()

    X_test = test[
        features
    ].copy()

    y_train = encoder.transform(
        train[
            TARGET_COLUMN
        ].astype(str)
    )

    y_test = encoder.transform(
        test[
            TARGET_COLUMN
        ].astype(str)
    )

    model = build_model(
        algorithm=algorithm,
        params=params,
        n_classes=len(
            encoder.classes_
        ),
        seed=stable_seed(
            f"outer-final|{algorithm}|"
            f"{outer_fold}"
        ),
    )

    fit_model(
        model,
        algorithm,
        X_train,
        y_train,
    )

    pred = model.predict(
        X_test
    )

    proba = model.predict_proba(
        X_test
    )

    try:
        auc = float(
            roc_auc_score(
                y_test,
                proba,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        auc = float(
            "nan"
        )

    return {
        "macro_f1": float(
            f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
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
        "macro_precision": float(
            precision_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc_macro_ovr": auc,
    }


def serialize_params(
    params: dict,
) -> str:

    return json.dumps(
        params,
        sort_keys=True,
        ensure_ascii=False,
    )


def main() -> None:
    print("=" * 100)
    print(
        "CYBELL-J — HIPERPARAMETRIZAÇÃO "
        "DO MODELO GLOBAL DE TECLADO"
    )
    print("=" * 100)

    print("\nProtocolo:")
    print("  somente teclado")
    print("  janela = 50 eventos")
    print("  overlap = 75%")
    print("  imputação = mediana")
    print("  scaling = nenhum")
    print(
        "  outer CV = 3 folds por session_id"
    )
    print(
        "  inner CV = 2 folds por session_id"
    )
    print(
        "  seleção = Macro F1"
    )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado:\n"
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        low_memory=False,
    )

    features = load_features()

    missing = [
        c
        for c in features
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Features ausentes no dataset: "
            + ", ".join(missing)
        )

    df = df[
        df["window_size"].eq(
            WINDOW_SIZE
        )
        & df[
            "requested_overlap_pct"
        ].eq(
            OVERLAP_PCT
        )
    ].copy().reset_index(
        drop=True
    )

    if df.empty:
        raise RuntimeError(
            "Nenhuma janela de teclado "
            "50/75 encontrada."
        )

    users = sorted(
        df[
            TARGET_COLUMN
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(users) != 6:
        raise RuntimeError(
            f"Esperadas 6 usuárias; "
            f"encontradas {len(users)}."
        )

    print(
        f"\nJanelas: "
        f"{len(df):,}".replace(
            ",",
            ".",
        )
    )
    print(
        f"Usuárias: {len(users)}"
    )
    print(
        f"Features: {len(features)}"
    )

    if len(features) != 59:
        print(
            f"AVISO: eram esperadas 59 features; "
            f"foram encontradas {len(features)}."
        )

    print(
        "\nSessões por usuária:"
    )

    print(
        df.groupby(
            "user"
        )[
            "session_id"
        ]
        .nunique()
        .sort_index()
        .to_string()
    )

    encoder = LabelEncoder()
    encoder.fit(
        users
    )

    outer_folds = build_session_folds(
        df,
        n_folds=OUTER_FOLDS,
        seed_prefix="outer",
    )

    algorithms = {
        "RandomForest": RF_CANDIDATES,
        "XGBoost": XGB_CANDIDATES,
    }

    inner_rows = []
    outer_rows = []

    for algorithm, candidates in algorithms.items():

        print(
            "\n"
            + "=" * 100
        )
        print(
            algorithm
        )
        print(
            "=" * 100
        )

        for outer_info in outer_folds:

            outer_fold = int(
                outer_info["fold"]
            )

            outer_train = df[
                df[
                    "session_id"
                ]
                .astype(str)
                .isin(
                    outer_info[
                        "train_sessions"
                    ]
                )
            ].copy()

            outer_test = df[
                df[
                    "session_id"
                ]
                .astype(str)
                .isin(
                    outer_info[
                        "test_sessions"
                    ]
                )
            ].copy()

            print(
                f"\nOuter fold {outer_fold}: "
                f"treino={len(outer_train):,} | "
                f"teste={len(outer_test):,}"
                .replace(
                    ",",
                    ".",
                )
            )

            candidate_results = []

            for candidate_id, params in enumerate(
                candidates,
                start=1,
            ):

                (
                    mean_score,
                    inner_scores,
                ) = evaluate_candidate_inner(
                    outer_train=outer_train,
                    features=features,
                    encoder=encoder,
                    algorithm=algorithm,
                    params=params,
                    outer_fold=outer_fold,
                    candidate_id=candidate_id,
                )

                candidate_results.append(
                    (
                        candidate_id,
                        params,
                        mean_score,
                    )
                )

                inner_rows.append(
                    {
                        "algorithm": algorithm,
                        "outer_fold": outer_fold,
                        "candidate_id": candidate_id,
                        "params_json": (
                            serialize_params(
                                params
                            )
                        ),
                        "inner_macro_f1_mean": (
                            mean_score
                        ),
                        "inner_fold_scores": (
                            "|".join(
                                f"{x:.6f}"
                                for x in inner_scores
                            )
                        ),
                    }
                )

                print(
                    f"  candidato "
                    f"{candidate_id}: "
                    f"MacroF1 inner="
                    f"{mean_score:.4f}"
                )

            candidate_results.sort(
                key=lambda x: x[2],
                reverse=True,
            )

            (
                best_id,
                best_params,
                best_inner,
            ) = candidate_results[0]

            metrics = evaluate_outer(
                train=outer_train,
                test=outer_test,
                features=features,
                encoder=encoder,
                algorithm=algorithm,
                params=best_params,
                outer_fold=outer_fold,
            )

            outer_rows.append(
                {
                    "algorithm": algorithm,
                    "outer_fold": outer_fold,
                    "selected_candidate_id": (
                        best_id
                    ),
                    "selected_params_json": (
                        serialize_params(
                            best_params
                        )
                    ),
                    "inner_macro_f1_selected": (
                        best_inner
                    ),
                    **metrics,
                }
            )

            print(
                f"  MELHOR inner: "
                f"candidato {best_id} "
                f"({best_inner:.4f})"
            )

            print(
                f"  TESTE externo: "
                f"MacroF1="
                f"{metrics['macro_f1']:.4f} | "
                f"BalAcc="
                f"{metrics['balanced_accuracy']:.4f} | "
                f"Acc="
                f"{metrics['accuracy']:.4f} | "
                f"AUC="
                f"{metrics['roc_auc_macro_ovr']:.4f}"
            )

    inner_df = pd.DataFrame(
        inner_rows
    )

    outer_df = pd.DataFrame(
        outer_rows
    )

    inner_df.to_csv(
        INNER_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    outer_df.to_csv(
        OUTER_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    best_payload = {}

    for algorithm in algorithms:

        sub = inner_df[
            inner_df[
                "algorithm"
            ].eq(
                algorithm
            )
        ].copy()

        candidate_summary = (
            sub.groupby(
                [
                    "candidate_id",
                    "params_json",
                ],
                as_index=False,
            )
            .agg(
                inner_macro_f1_mean=(
                    "inner_macro_f1_mean",
                    "mean",
                ),
                inner_macro_f1_std=(
                    "inner_macro_f1_mean",
                    lambda s: float(
                        np.std(
                            s.to_numpy(
                                dtype=float
                            ),
                            ddof=0,
                        )
                    ),
                ),
            )
            .sort_values(
                [
                    "inner_macro_f1_mean",
                    "inner_macro_f1_std",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        candidate_summary.insert(
            0,
            "algorithm",
            algorithm,
        )

        summary_rows.append(
            candidate_summary
        )

        best_candidate = (
            candidate_summary.iloc[0]
        )

        outer_sub = outer_df[
            outer_df[
                "algorithm"
            ].eq(
                algorithm
            )
        ]

        best_payload[
            algorithm
        ] = {
            "recommended_candidate_id": int(
                best_candidate[
                    "candidate_id"
                ]
            ),
            "recommended_params": json.loads(
                best_candidate[
                    "params_json"
                ]
            ),
            "inner_macro_f1_mean": float(
                best_candidate[
                    "inner_macro_f1_mean"
                ]
            ),
            "inner_macro_f1_std": float(
                best_candidate[
                    "inner_macro_f1_std"
                ]
            ),
            "nested_outer_metrics": {
                "macro_f1_mean": float(
                    outer_sub[
                        "macro_f1"
                    ].mean()
                ),
                "balanced_accuracy_mean": float(
                    outer_sub[
                        "balanced_accuracy"
                    ].mean()
                ),
                "accuracy_mean": float(
                    outer_sub[
                        "accuracy"
                    ].mean()
                ),
                "roc_auc_macro_ovr_mean": float(
                    outer_sub[
                        "roc_auc_macro_ovr"
                    ].mean()
                ),
            },
        }

    summary_df = pd.concat(
        summary_rows,
        ignore_index=True,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "experiment": (
            "Hiperparametrização global de teclado"
        ),
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP_PCT,
        "features": len(
            features
        ),
        "normalization_preprocessing": {
            "imputation": (
                "SimpleImputer(strategy='median')"
            ),
            "scaling": "none",
        },
        "cross_validation": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "grouping": "session_id",
            "selection_metric": "macro_f1",
        },
        "recommended": (
            best_payload
        ),
    }

    with BEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "HIPERPARAMETRIZAÇÃO "
        "DO TECLADO CONCLUÍDA"
    )
    print(
        "=" * 100
    )

    for (
        algorithm,
        result,
    ) in best_payload.items():

        print(
            f"\n{algorithm}"
        )

        print(
            "  parâmetros recomendados: "
            + json.dumps(
                result[
                    "recommended_params"
                ],
                ensure_ascii=False,
            )
        )

        print(
            f"  MacroF1 inner médio: "
            f"{result['inner_macro_f1_mean']:.4f}"
        )

        print(
            f"  MacroF1 outer médio: "
            f"{result['nested_outer_metrics']['macro_f1_mean']:.4f}"
        )

        print(
            f"  Balanced Accuracy "
            f"outer média: "
            f"{result['nested_outer_metrics']['balanced_accuracy_mean']:.4f}"
        )

        print(
            f"  Accuracy outer média: "
            f"{result['nested_outer_metrics']['accuracy_mean']:.4f}"
        )

    print(
        "\nArquivos gerados:"
    )
    print(
        f"  {INNER_PATH}"
    )
    print(
        f"  {OUTER_PATH}"
    )
    print(
        f"  {SUMMARY_PATH}"
    )
    print(
        f"  {BEST_PATH}"
    )

    print(
        "\nNenhum resultado anterior "
        "foi sobrescrito."
    )


if __name__ == "__main__":
    main()
