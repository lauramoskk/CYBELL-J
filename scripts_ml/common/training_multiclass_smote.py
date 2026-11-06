from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from xgboost import XGBClassifier

from .training_multiclass import (
    RANDOM_STATE,
    RF_GRID,
    XGB_GRID,
    calculate_metrics,
    mean_metrics,
)


def create_model_smote(algorithm):

    if algorithm == "RandomForest":
        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        grid = RF_GRID

    elif algorithm == "XGBoost":
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        grid = XGB_GRID

    else:
        raise ValueError(
            f"Algoritmo desconhecido: {algorithm}"
        )

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "smote",
            SMOTE(
                sampling_strategy="auto",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "model",
            model,
        ),
    ])

    return pipeline, grid


def nested_cv_smote(
    X,
    y,
    groups,
    algorithm,
):

    pipeline, grid = create_model_smote(
        algorithm
    )

    outer_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    oof_rows = []
    balance_rows = []

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

        train_counts = (
            pd.Series(y_train)
            .value_counts()
            .sort_index()
        )

        majority_count = int(
            train_counts.max()
        )

        for class_id, count in (
            train_counts.items()
        ):
            balance_rows.append(
                {
                    "fold": fold,
                    "class_id":
                        int(class_id),
                    "before_smote":
                        int(count),
                    "after_smote":
                        majority_count,
                }
            )

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

        search.fit(
            X_train,
            y_train,
            groups=groups_train,
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
            f"{algorithm} | "
            f"fold {fold} | "
            f"Macro F1="
            f"{metrics['macro_f1']:.4f} | "
            f"Acc="
            f"{metrics['accuracy']:.4f} | "
            f"BalAcc="
            f"{metrics['balanced_accuracy']:.4f}"
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

    balance_df = pd.DataFrame(
        balance_rows
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

    final_search.fit(
        X,
        y,
        groups=groups,
    )

    return (
        folds_df,
        oof_df,
        balance_df,
        final_search,
    )


def run_multiclass_smote(
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

    groups = (
        df["user"].astype(str)
        + "::"
        + df[
            "session_id"
        ].astype(str)
    )

    print("\nDistribuição ORIGINAL:")
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
        f"\nJanelas originais: "
        f"{len(df)}"
    )

    print(
        f"Features: {len(features)}"
    )

    experiments = {}
    balance_saved = False

    for algorithm in [
        "RandomForest",
        "XGBoost",
    ]:

        print(
            "\n" + "=" * 70
        )

        print(
            f"{algorithm} + SMOTE"
        )

        print("=" * 70)

        (
            folds,
            oof,
            balance,
            search,
        ) = nested_cv_smote(
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

        if not balance_saved:

            balance = balance.copy()

            balance[
                "user"
            ] = balance[
                "class_id"
            ].map(
                lambda value:
                encoder.inverse_transform(
                    [value]
                )[0]
            )

            balance.to_csv(
                output_dir
                / "balanceamento_folds.csv",
                index=False,
            )

            balance_saved = True

    result_rows = []

    for algorithm, result in (
        experiments.items()
    ):

        result_rows.append(
            {
                "algorithm":
                    algorithm,
                **result[
                    "metrics"
                ],
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
        results_df.iloc[
            0
        ]["algorithm"]
    )

    best = experiments[
        best_algorithm
    ]

    oof = best[
        "oof"
    ]

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
        target_names=
            encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    user_rows = []

    for user in (
        encoder.classes_
    ):

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
            (
                "SimpleImputer(median) "
                "+ StandardScaler"
            ),
        "balancing":
            (
                "SMOTE("
                "sampling_strategy=auto, "
                "random_state=42)"
            ),
        "n_features":
            len(features),
    }

    joblib.dump(
        bundle,
        output_dir
        / (
            "model_global_"
            "6_users_smote.joblib"
        ),
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
            "global_6_users_smote",
        "modality":
            modality,
        "users":
            list(
                encoder.classes_
            ),
        "n_original_windows":
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
            (
                "SimpleImputer(median) "
                "+ StandardScaler"
            ),
        "balancing":
            (
                "SMOTE inside training "
                "folds only"
            ),
        "sampling_strategy":
            "auto",
        "outer_cv":
            (
                "StratifiedGroupKFold"
                "(n_splits=3)"
            ),
        "inner_cv":
            (
                "StratifiedGroupKFold"
                "(n_splits=2)"
            ),
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
        "windows_per_user_original":
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
        "RESULTADO FINAL COM SMOTE"
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
        "\nModelo salvo em:"
    )

    print(
        output_dir
        / (
            "model_global_"
            "6_users_smote.joblib"
        )
    )