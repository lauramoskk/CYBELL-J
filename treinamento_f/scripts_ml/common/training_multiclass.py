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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        return build_pipeline(model), XGB_GRID

    raise ValueError(
        f"Algoritmo desconhecido: {algorithm}"
    )


def calculate_metrics(y_true, proba):

    pred = np.argmax(
        proba,
        axis=1,
    )

    try:
        auc = roc_auc_score(
            y_true,
            proba,
            multi_class="ovr",
            average="macro",
            labels=np.arange(
                proba.shape[1]
            ),
        )
    except ValueError:
        auc = np.nan

    return {
        "accuracy": accuracy_score(
            y_true,
            pred,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                pred,
            ),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0,
        ),
        "macro_precision":
            precision_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            ),
        "macro_recall":
            recall_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            ),
        "roc_auc_ovr_macro":
            auc,
    }


def nested_cv(
    X,
    y,
    groups,
    algorithm,
):

    pipeline, grid = create_model(
        algorithm
    )

    outer_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    oof_rows = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer_cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        groups_train = groups.iloc[
            train_idx
        ]

        inner_cv = StratifiedGroupKFold(
            n_splits=2,
            shuffle=True,
            random_state=(
                RANDOM_STATE + fold
            ),
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )

        weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        search.fit(
            X_train,
            y_train,
            groups=groups_train,
            model__sample_weight=weights,
        )

        proba = (
            search
            .best_estimator_
            .predict_proba(
                X_test
            )
        )

        pred = np.argmax(
            proba,
            axis=1,
        )

        metrics = calculate_metrics(
            y_test,
            proba,
        )

        fold_rows.append(
            {
                "fold": fold,
                "inner_macro_f1":
                    float(
                        search.best_score_
                    ),
                "best_params":
                    json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    ),
                **metrics,
            }
        )

        for position, row_index in enumerate(
            test_idx
        ):
            oof_rows.append(
                {
                    "row_index":
                        int(row_index),
                    "fold":
                        fold,
                    "y_true":
                        int(
                            y_test[
                                position
                            ]
                        ),
                    "y_pred":
                        int(
                            pred[
                                position
                            ]
                        ),
                }
            )

        print(
            f"{algorithm} | fold {fold} | "
            f"Macro F1={metrics['macro_f1']:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"BalAcc={metrics['balanced_accuracy']:.4f}"
        )

    folds_df = pd.DataFrame(
        fold_rows
    )

    oof_df = (
        pd.DataFrame(
            oof_rows
        )
        .sort_values(
            "row_index"
        )
        .reset_index(
            drop=True
        )
    )

    final_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    final_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1_macro",
        cv=final_cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
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


def mean_metrics(folds):

    columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "roc_auc_ovr_macro",
    ]

    return {
        column:
            float(
                folds[
                    column
                ].mean()
            )
        for column in columns
    }


def run_multiclass(
    df,
    features,
    users,
    output_dir,
    modality,
    window_size,
    overlap,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    found = set(
        df["user"].unique()
    )

    missing = [
        user
        for user in users
        if user not in found
    ]

    if missing:
        raise RuntimeError(
            "Usuárias ausentes no dataset: "
            + ", ".join(missing)
        )

    df = (
        df[
            df["user"].isin(
                users
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    encoder = LabelEncoder()
    encoder.fit(users)

    y = encoder.transform(
        df["user"]
    )

    X = df[
        features
    ]

    # Evita colisão caso session_id
    # se repita entre pessoas.
    groups = (
        df["user"].astype(str)
        + "::"
        + df["session_id"].astype(str)
    )

    print("\nParticipantes:")
    print(
        df["user"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSessões:")
    print(
        df.groupby(
            "user"
        )["session_id"]
        .nunique()
        .sort_index()
        .to_string()
    )

    print(
        f"\nJanelas: {len(df)}"
    )

    print(
        f"Features: {len(features)}"
    )

    experiments = {}

    for algorithm in [
        "RandomForest",
        "XGBoost",
    ]:

        print(
            "\n" + "=" * 70
        )
        print(algorithm)
        print("=" * 70)

        folds, oof, search = nested_cv(
            X,
            y,
            groups,
            algorithm,
        )

        metrics = mean_metrics(
            folds
        )

        experiments[
            algorithm
        ] = {
            "folds": folds,
            "oof": oof,
            "search": search,
            "metrics": metrics,
        }

        folds.to_csv(
            output_dir
            / f"{algorithm}_folds.csv",
            index=False,
        )

    result_rows = []

    for algorithm, result in (
        experiments.items()
    ):

        result_rows.append(
            {
                "algorithm":
                    algorithm,
                **result["metrics"],
            }
        )

    results_df = (
        pd.DataFrame(
            result_rows
        )
        .sort_values(
            "macro_f1",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    results_df.to_csv(
        output_dir
        / "resultados_modelos.csv",
        index=False,
    )

    best_algorithm = (
        results_df.iloc[0][
            "algorithm"
        ]
    )

    best = experiments[
        best_algorithm
    ]

    oof = best["oof"]

    cm = confusion_matrix(
        oof["y_true"],
        oof["y_pred"],
        labels=np.arange(
            len(
                encoder.classes_
            )
        ),
    )

    cm_df = pd.DataFrame(
        cm,
        columns=[
            f"pred_{name}"
            for name
            in encoder.classes_
        ],
    )

    cm_df.insert(
        0,
        "actual_user",
        encoder.classes_,
    )

    cm_df.to_csv(
        output_dir
        / "confusion_matrix.csv",
        index=False,
    )

    report = classification_report(
        oof["y_true"],
        oof["y_pred"],
        labels=np.arange(
            len(
                encoder.classes_
            )
        ),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    user_rows = []

    for user in encoder.classes_:

        values = report[
            user
        ]

        user_rows.append(
            {
                "user": user,
                "precision":
                    values[
                        "precision"
                    ],
                "recall":
                    values[
                        "recall"
                    ],
                "f1":
                    values[
                        "f1-score"
                    ],
                "support":
                    int(
                        values[
                            "support"
                        ]
                    ),
            }
        )

    pd.DataFrame(
        user_rows
    ).to_csv(
        output_dir
        / "metricas_por_usuario.csv",
        index=False,
    )

    final_search = best[
        "search"
    ]

    bundle = {
        "pipeline":
            final_search
            .best_estimator_,
        "features":
            features,
        "label_encoder":
            encoder,
        "classes":
            list(
                encoder.classes_
            ),
        "best_algorithm":
            best_algorithm,
        "best_params":
            final_search
            .best_params_,
        "window_size":
            window_size,
        "overlap_pct":
            overlap,
        "modality":
            modality,
        "normalization":
            "SimpleImputer(median) + StandardScaler",
        "n_features":
            len(features),
    }

    joblib.dump(
        bundle,
        output_dir
        / "model_global_6_users.joblib",
    )

    windows_per_user = {
        str(key):
            int(value)
        for key, value
        in df[
            "user"
        ]
        .value_counts()
        .sort_index()
        .items()
    }

    sessions_per_user = {
        str(key):
            int(value)
        for key, value
        in df.groupby(
            "user"
        )["session_id"]
        .nunique()
        .sort_index()
        .items()
    }

    summary = {
        "experiment":
            "global_6_users",
        "modality":
            modality,
        "users":
            list(
                encoder.classes_
            ),
        "n_windows":
            int(
                len(df)
            ),
        "n_features":
            int(
                len(features)
            ),
        "window_size":
            window_size,
        "overlap_pct":
            overlap,
        "normalization":
            "SimpleImputer(median) + StandardScaler",
        "outer_cv":
            "StratifiedGroupKFold(n_splits=3)",
        "inner_cv":
            "StratifiedGroupKFold(n_splits=2)",
        "selection_metric":
            "macro_f1",
        "best_algorithm":
            best_algorithm,
        "best_params":
            final_search
            .best_params_,
        "outer_cv_metrics":
            best[
                "metrics"
            ],
        "windows_per_user":
            windows_per_user,
        "sessions_per_user":
            sessions_per_user,
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
        "\n" + "=" * 70
    )

    print(
        "RESULTADO FINAL"
    )

    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nMelhor algoritmo:",
        best_algorithm,
    )

    print(
        "Melhores hiperparâmetros:"
    )

    print(
        final_search.best_params_
    )

    print(
        "\nModelo salvo em:"
    )

    print(
        output_dir
        / "model_global_6_users.joblib"
    )