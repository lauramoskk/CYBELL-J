from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from .preprocessing import build_pipeline


RANDOM_STATE = 42


# ---------------------------------------------------------
# HIPERPARÂMETROS
# Busca pequena e controlada.
# ---------------------------------------------------------

RF_GRID = [
    {
        "model__n_estimators": [300],
        "model__max_depth": [None],
        "model__min_samples_leaf": [1],
        "model__max_features": ["sqrt"],
    },
    {
        "model__n_estimators": [400],
        "model__max_depth": [None],
        "model__min_samples_leaf": [2],
        "model__max_features": [0.7],
    },
    {
        "model__n_estimators": [500],
        "model__max_depth": [20],
        "model__min_samples_leaf": [2],
        "model__max_features": ["sqrt"],
    },
    {
        "model__n_estimators": [500],
        "model__max_depth": [None],
        "model__min_samples_leaf": [4],
        "model__max_features": [0.7],
    },
]


XGB_GRID = [
    {
        "model__n_estimators": [250],
        "model__learning_rate": [0.05],
        "model__max_depth": [4],
        "model__min_child_weight": [1],
        "model__subsample": [0.85],
        "model__colsample_bytree": [0.85],
    },
    {
        "model__n_estimators": [400],
        "model__learning_rate": [0.05],
        "model__max_depth": [4],
        "model__min_child_weight": [2],
        "model__subsample": [0.90],
        "model__colsample_bytree": [0.90],
    },
    {
        "model__n_estimators": [350],
        "model__learning_rate": [0.05],
        "model__max_depth": [6],
        "model__min_child_weight": [2],
        "model__subsample": [0.85],
        "model__colsample_bytree": [0.85],
    },
    {
        "model__n_estimators": [300],
        "model__learning_rate": [0.10],
        "model__max_depth": [4],
        "model__min_child_weight": [1],
        "model__subsample": [0.90],
        "model__colsample_bytree": [0.80],
    },
]


def create_model(algorithm):

    if algorithm == "RandomForest":

        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        return build_pipeline(model), RF_GRID

    if algorithm == "XGBoost":

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

        return build_pipeline(model), XGB_GRID

    raise ValueError(
        f"Algoritmo desconhecido: {algorithm}"
    )


# ---------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),

        "macro_f1":
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            ),

        "auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),
    }


def calculate_eer(
    y_true,
    probabilities,
):

    fpr, tpr, thresholds = (
        roc_curve(
            y_true,
            probabilities,
        )
    )

    fnr = 1.0 - tpr

    index = np.argmin(
        np.abs(
            fpr - fnr
        )
    )

    eer = (
        fpr[index]
        + fnr[index]
    ) / 2.0

    return (
        float(eer),
        float(
            thresholds[index]
        ),
    )


# ---------------------------------------------------------
# CROSS VALIDATION + HIPERPARAMETRIZAÇÃO
# ---------------------------------------------------------

def nested_cv(
    X,
    y,
    groups,
    algorithm,
):

    pipeline, grid = (
        create_model(
            algorithm
        )
    )

    outer_cv = (
        StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    )

    fold_results = []
    oof_predictions = []

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        outer_cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y[
            train_index
        ]

        y_test = y[
            test_index
        ]

        groups_train = groups.iloc[
            train_index
        ]

        # -------------------------------
        # HIPERPARAMETRIZAÇÃO
        # somente dentro do treino
        # -------------------------------

        inner_cv = (
            StratifiedGroupKFold(
                n_splits=2,
                shuffle=True,
                random_state=(
                    RANDOM_STATE
                    + fold
                ),
            )
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
        )

        weights = (
            compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
        )

        search.fit(
            X_train,
            y_train,
            groups=groups_train,
            model__sample_weight=weights,
        )

        probabilities = (
            search
            .best_estimator_
            .predict_proba(
                X_test
            )[:, 1]
        )

        metrics = (
            calculate_metrics(
                y_test,
                probabilities,
            )
        )

        fold_results.append(
            {
                "fold": fold,

                "inner_macro_f1":
                    search.best_score_,

                "best_params":
                    json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    ),

                **metrics,
            }
        )

        for position, index in enumerate(
            test_index
        ):

            oof_predictions.append(
                {
                    "row_index":
                        int(index),

                    "fold":
                        fold,

                    "y_true":
                        int(
                            y[index]
                        ),

                    "probability":
                        float(
                            probabilities[
                                position
                            ]
                        ),
                }
            )

    folds_df = pd.DataFrame(
        fold_results
    )

    oof_df = (
        pd.DataFrame(
            oof_predictions
        )
        .sort_values(
            "row_index"
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------
    # Hiperparametrização final
    # usando todos os dados conhecidos
    # ---------------------------------

    final_cv = (
        StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    )

    final_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1_macro",
        cv=final_cv,
        n_jobs=1,
        refit=True,
    )

    weights = (
        compute_sample_weight(
            class_weight="balanced",
            y=y,
        )
    )

    final_search.fit(
        X,
        y,
        groups=groups,
        model__sample_weight=weights,
    )

    return (
        folds_df,
        oof_df,
        final_search,
    )


def mean_metrics(
    folds_df,
):

    columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "precision",
        "recall",
        "auc",
    ]

    return {
        column:
            float(
                folds_df[
                    column
                ].mean()
            )
        for column in columns
    }


# ---------------------------------------------------------
# GLOBAL YASS x LINA
# ---------------------------------------------------------

def run_global(
    df,
    features,
    output_dir,
    modality,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    known_users = [
        "yass",
        "lina",
    ]

    df = df[
        df["user"]
        .isin(
            known_users
        )
    ].copy().reset_index(
        drop=True
    )

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        df["user"]
    )

    X = df[
        features
    ]

    groups = df[
        "session_id"
    ].astype(str)

    algorithms = [
        "RandomForest",
        "XGBoost",
    ]

    experiments = {}

    for algorithm in algorithms:

        print(
            f"\n{algorithm}"
        )

        folds, oof, search = (
            nested_cv(
                X,
                y,
                groups,
                algorithm,
            )
        )

        metrics = mean_metrics(
            folds
        )

        experiments[
            algorithm
        ] = {
            "folds":
                folds,

            "oof":
                oof,

            "search":
                search,

            "metrics":
                metrics,
        }

        folds.to_csv(
            output_dir
            / f"{algorithm}_folds.csv",
            index=False,
        )

        print(
            metrics
        )

    best_algorithm = max(
        experiments,
        key=lambda name:
            experiments[name][
                "metrics"
            ][
                "macro_f1"
            ],
    )

    best = experiments[
        best_algorithm
    ]

    oof = best["oof"]

    predictions = (
        oof["probability"]
        >= 0.5
    ).astype(int)

    correct = oof[
        predictions
        == oof["y_true"]
    ].copy()

    confidence = np.maximum(
        correct["probability"],
        1
        - correct["probability"],
    )

    # Threshold exploratório para
    # reconhecer UNKNOWN posteriormente.
    unknown_threshold = float(
        confidence.quantile(
            0.05
        )
    )

    bundle = {
        "pipeline":
            best["search"]
            .best_estimator_,

        "features":
            features,

        "label_encoder":
            encoder,

        "known_users":
            known_users,

        "unknown_threshold":
            unknown_threshold,

        "best_algorithm":
            best_algorithm,

        "best_params":
            best["search"]
            .best_params_,

        "modality":
            modality,

        "normalization":
            "median + StandardScaler",
    }

    joblib.dump(
        bundle,
        output_dir
        / "model_global.joblib",
    )

    summary = {
        "best_algorithm":
            best_algorithm,

        "best_params":
            best["search"]
            .best_params_,

        "metrics":
            best["metrics"],

        "unknown_threshold":
            unknown_threshold,
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\nMelhor modelo:",
        best_algorithm,
    )

    print(
        "Threshold UNKNOWN:",
        round(
            unknown_threshold,
            4,
        ),
    )


# ---------------------------------------------------------
# INDIVIDUAL
# ---------------------------------------------------------

def run_individual(
    df,
    features,
    output_dir,
    modality,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    impostors = [
        "Julia104soup",
        "bianca",
        "lala",
        "laura_oliveira",
    ]

    configs = {
        "yass": {
            "algorithm":
                "RandomForest",

            "excluded":
                "lina",
        },

        "lina": {
            "algorithm":
                "XGBoost",

            "excluded":
                "yass",
        },
    }

    for target, config in (
        configs.items()
    ):

        allowed_users = (
            [target]
            + impostors
        )

        subset = df[
            df["user"]
            .isin(
                allowed_users
            )
        ].copy().reset_index(
            drop=True
        )

        # A pessoa cruzada não aparece
        # em nenhuma parte do treino.
        assert (
            config["excluded"]
            not in subset[
                "user"
            ].unique()
        )

        X = subset[
            features
        ]

        y = (
            subset["user"]
            .eq(target)
            .astype(int)
            .to_numpy()
        )

        groups = (
            subset[
                "session_id"
            ].astype(str)
        )

        folds, oof, search = (
            nested_cv(
                X,
                y,
                groups,
                config[
                    "algorithm"
                ],
            )
        )

        metrics = mean_metrics(
            folds
        )

        eer, threshold = (
            calculate_eer(
                oof["y_true"],
                oof[
                    "probability"
                ],
            )
        )

        bundle = {
            "pipeline":
                search.best_estimator_,

            "features":
                features,

            "target":
                target,

            "algorithm":
                config[
                    "algorithm"
                ],

            "known_impostors":
                impostors,

            "excluded_unknown_user":
                config[
                    "excluded"
                ],

            "threshold":
                threshold,

            "eer":
                eer,

            "best_params":
                search.best_params_,

            "metrics":
                metrics,

            "modality":
                modality,

            "normalization":
                "median + StandardScaler",
        }

        joblib.dump(
            bundle,
            output_dir
            / f"{target}.joblib",
        )

        folds.to_csv(
            output_dir
            / f"{target}_folds.csv",
            index=False,
        )

        oof.to_csv(
            output_dir
            / f"{target}_oof.csv",
            index=False,
        )

        with (
            output_dir
            / f"{target}_summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "metrics":
                        metrics,

                    "eer":
                        eer,

                    "threshold":
                        threshold,

                    "best_params":
                        search.best_params_,

                    "excluded_unknown_user":
                        config[
                            "excluded"
                        ],
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\n{target.upper()}"
        )

        print(
            "Algoritmo:",
            config[
                "algorithm"
            ],
        )

        print(
            "Desconhecida reservada:",
            config[
                "excluded"
            ],
        )

        print(
            "EER:",
            round(
                eer,
                4,
            ),
        )

        print(
            "Métricas:",
            metrics,
        )