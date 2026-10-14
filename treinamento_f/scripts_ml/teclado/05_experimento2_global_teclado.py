"""
05_experimento2_global_teclado.py

CYBELL-J — Experimento 2
RECONHECIMENTO GLOBAL MULTICLASSE USANDO APENAS TECLADO

Replica o protocolo do experimento global de mouse:
- 6 usuárias;
- 12 configurações: janelas 25/30/50/100 x overlaps 25/50/75%;
- Random Forest e XGBoost;
- 3 folds por session_id;
- nenhuma sessão é dividida entre treino e teste;
- SimpleImputer(strategy="median") dentro do pipeline;
- seleção principal pelo Macro F1.

Entrada:
data/teclado_dataset/dataset_teclado.csv
data/teclado_dataset/dataset_teclado_manifest.json

Saídas:
data/teclado_experimento2/
    resultados_folds.csv
    resumo_configuracoes.csv
    predicoes_oof.csv
    eer_por_usuario.csv
    matriz_confusao_melhor.csv
    metricas_por_usuario_melhor.csv
    melhor_modelo_identificacao.joblib
    melhor_modelo_features.json
    manifesto_experimento2.json
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "ERRO: xgboost não está instalado neste ambiente."
    ) from exc


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "teclado_dataset"
    / "dataset_teclado.csv"
)

MANIFEST_DATASET_PATH = (
    BASE_DIR
    / "data"
    / "teclado_dataset"
    / "dataset_teclado_manifest.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "teclado_experimento2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS_PATH = OUTPUT_DIR / "resultados_folds.csv"
SUMMARY_PATH = OUTPUT_DIR / "resumo_configuracoes.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predicoes_oof.csv"
EER_USERS_PATH = OUTPUT_DIR / "eer_por_usuario.csv"
CONFUSION_PATH = OUTPUT_DIR / "matriz_confusao_melhor.csv"
USER_METRICS_PATH = OUTPUT_DIR / "metricas_por_usuario_melhor.csv"
MODEL_PATH = OUTPUT_DIR / "melhor_modelo_identificacao.joblib"
FEATURES_PATH = OUTPUT_DIR / "melhor_modelo_features.json"
MANIFEST_PATH = OUTPUT_DIR / "manifesto_experimento2.json"

TARGET_COLUMN = "user"

WINDOW_SIZES = [25, 30, 50, 100]
OVERLAPS = [25, 50, 75]

N_FOLDS = 3
RANDOM_STATE = 42


def stable_seed_for_user(username: str) -> int:
    digest = hashlib.sha256(
        username.encode("utf-8")
    ).hexdigest()

    return (
        RANDOM_STATE
        + int(digest[:8], 16)
    ) % (2**32 - 1)


def load_feature_contract() -> list[str]:
    if not MANIFEST_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Manifesto do dataset não encontrado:\n"
            f"{MANIFEST_DATASET_PATH}"
        )

    with MANIFEST_DATASET_PATH.open(
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
            "O manifesto não contém feature_columns."
        )

    return list(features)


def validate_features(
    df: pd.DataFrame,
    features: list[str],
) -> None:
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

    non_numeric = [
        c
        for c in features
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if non_numeric:
        raise RuntimeError(
            "Features não numéricas encontradas: "
            + ", ".join(non_numeric)
        )


def build_session_folds(
    df_config: pd.DataFrame,
    n_folds: int = N_FOLDS,
) -> list[dict]:

    users = sorted(
        df_config[TARGET_COLUMN]
        .astype(str)
        .unique()
        .tolist()
    )

    user_sessions = {}

    for user in users:
        sessions = sorted(
            df_config.loc[
                df_config[TARGET_COLUMN]
                .astype(str)
                .eq(user),
                "session_id",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        user_sessions[user] = sessions

    too_few = {
        user: len(sessions)
        for user, sessions in user_sessions.items()
        if len(sessions) < n_folds
    }

    if too_few:
        raise RuntimeError(
            "Sessões insuficientes para 3 folds: "
            f"{too_few}"
        )

    fold_test_sessions = [
        []
        for _ in range(n_folds)
    ]

    for user in users:
        sessions = list(
            user_sessions[user]
        )

        rng = np.random.default_rng(
            stable_seed_for_user(user)
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
    seen_test = []

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

        train = df_config[
            df_config["session_id"]
            .astype(str)
            .isin(train_sessions)
        ]

        test = df_config[
            df_config["session_id"]
            .astype(str)
            .isin(test_sessions)
        ]

        needed_users = set(users)

        if set(
            train[TARGET_COLUMN]
            .astype(str)
            .unique()
        ) != needed_users:
            raise RuntimeError(
                f"Fold {fold_idx + 1}: treino sem todas as usuárias."
            )

        if set(
            test[TARGET_COLUMN]
            .astype(str)
            .unique()
        ) != needed_users:
            raise RuntimeError(
                f"Fold {fold_idx + 1}: teste sem todas as usuárias."
            )

        overlap = (
            set(train_sessions)
            & set(test_sessions)
        )

        if overlap:
            raise RuntimeError(
                f"Fold {fold_idx + 1}: vazamento de sessão."
            )

        seen_test.extend(test_sessions)

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_sessions": train_sessions,
                "test_sessions": test_sessions,
            }
        )

    if len(seen_test) != len(set(seen_test)):
        raise RuntimeError(
            "Uma sessão apareceu em teste em mais de um fold."
        )

    if set(seen_test) != all_sessions:
        raise RuntimeError(
            "Nem todas as sessões apareceram exatamente uma vez em teste."
        )

    return folds


def build_random_forest() -> Pipeline:
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
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgboost(
    n_classes: int,
) -> Pipeline:
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
                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=2,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def compute_binary_eer(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:

    y_true_binary = np.asarray(
        y_true_binary,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if len(
        np.unique(
            y_true_binary
        )
    ) < 2:
        return (
            float("nan"),
            float("nan"),
        )

    fpr, tpr, thresholds = roc_curve(
        y_true_binary,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    idx = int(
        np.nanargmin(
            np.abs(
                fpr - fnr
            )
        )
    )

    eer = float(
        (
            fpr[idx]
            + fnr[idx]
        )
        / 2.0
    )

    threshold = float(
        thresholds[idx]
    )

    return eer, threshold


def calculate_ovr_eer(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    label_encoder: LabelEncoder,
    model_name: str,
    window_size: int,
    overlap_pct: int,
    fold: int,
) -> tuple[float, list[dict]]:

    rows = []
    eers = []

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    for class_index, username in enumerate(classes):
        encoded_class = int(
            label_encoder.transform(
                [username]
            )[0]
        )

        binary_true = (
            y_true == encoded_class
        ).astype(int)

        score = probabilities[
            :,
            encoded_class
        ]

        eer, threshold = compute_binary_eer(
            binary_true,
            score,
        )

        eers.append(eer)

        rows.append(
            {
                "model": model_name,
                "window_size": window_size,
                "overlap_pct": overlap_pct,
                "fold": fold,
                "user": username,
                "eer_ovr": eer,
                "eer_threshold": threshold,
            }
        )

    return (
        float(
            np.nanmean(eers)
        ),
        rows,
    )


def train_one_fold(
    model_name: str,
    model: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    label_encoder: LabelEncoder,
    window_size: int,
    overlap_pct: int,
    fold_number: int,
) -> tuple[
    dict,
    pd.DataFrame,
    list[dict],
]:

    X_train = train[
        features
    ].copy()

    X_test = test[
        features
    ].copy()

    y_train = label_encoder.transform(
        train[
            TARGET_COLUMN
        ].astype(str)
    )

    y_test = label_encoder.transform(
        test[
            TARGET_COLUMN
        ].astype(str)
    )

    fit_kwargs = {}

    if model_name == "XGBoost":
        sample_weight = (
            compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )
        )

        fit_kwargs[
            "model__sample_weight"
        ] = sample_weight

    start_fit = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        **fit_kwargs,
    )

    fit_seconds = (
        time.perf_counter()
        - start_fit
    )

    start_predict = time.perf_counter()

    pred = model.predict(
        X_test
    )

    proba = model.predict_proba(
        X_test
    )

    predict_seconds = (
        time.perf_counter()
        - start_predict
    )

    macro_f1 = float(
        f1_score(
            y_test,
            pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            y_test,
            pred,
            average="weighted",
            zero_division=0,
        )
    )

    accuracy = float(
        accuracy_score(
            y_test,
            pred,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y_test,
            pred,
        )
    )

    macro_precision = float(
        precision_score(
            y_test,
            pred,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = float(
        recall_score(
            y_test,
            pred,
            average="macro",
            zero_division=0,
        )
    )

    try:
        auc_macro_ovr = float(
            roc_auc_score(
                y_test,
                proba,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        auc_macro_ovr = float("nan")

    users = (
        label_encoder
        .classes_
        .tolist()
    )

    macro_eer, eer_rows = calculate_ovr_eer(
        y_true=y_test,
        probabilities=proba,
        classes=users,
        label_encoder=label_encoder,
        model_name=model_name,
        window_size=window_size,
        overlap_pct=overlap_pct,
        fold=fold_number,
    )

    result = {
        "model": model_name,
        "window_size": int(window_size),
        "overlap_pct": int(overlap_pct),
        "fold": int(fold_number),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(features)),
        "n_users": int(len(users)),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "roc_auc_macro_ovr": auc_macro_ovr,
        "eer_macro_ovr_diagnostic": macro_eer,
        "fit_seconds": float(fit_seconds),
        "predict_ms_per_window": float(
            predict_seconds
            / max(len(test), 1)
            * 1000.0
        ),
        "train_sessions": "|".join(
            sorted(
                train["session_id"]
                .astype(str)
                .unique()
            )
        ),
        "test_sessions": "|".join(
            sorted(
                test["session_id"]
                .astype(str)
                .unique()
            )
        ),
    }

    prediction_data = {
        "model": np.repeat(
            model_name,
            len(test),
        ),
        "window_size": np.repeat(
            window_size,
            len(test),
        ),
        "overlap_pct": np.repeat(
            overlap_pct,
            len(test),
        ),
        "fold": np.repeat(
            fold_number,
            len(test),
        ),
        "session_id": (
            test["session_id"]
            .astype(str)
            .to_numpy()
        ),
        "window_id": (
            test["window_id"]
            .astype(str)
            .to_numpy()
        ),
        "y_true_encoded": y_test,
        "y_pred_encoded": pred,
        "y_true_user": (
            label_encoder
            .inverse_transform(
                y_test
            )
        ),
        "y_pred_user": (
            label_encoder
            .inverse_transform(
                pred.astype(int)
            )
        ),
    }

    for class_index, username in enumerate(users):
        prediction_data[
            f"proba__{username}"
        ] = proba[
            :,
            class_index
        ]

    return (
        result,
        pd.DataFrame(
            prediction_data
        ),
        eer_rows,
    )


def summarize_results(
    folds_df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "macro_f1",
        "weighted_f1",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "roc_auc_macro_ovr",
        "eer_macro_ovr_diagnostic",
        "fit_seconds",
        "predict_ms_per_window",
    ]

    rows = []

    grouped = folds_df.groupby(
        [
            "model",
            "window_size",
            "overlap_pct",
        ],
        dropna=False,
        sort=True,
    )

    for keys, group in grouped:
        model, window_size, overlap_pct = keys

        row = {
            "model": model,
            "window_size": int(window_size),
            "overlap_pct": int(overlap_pct),
            "n_folds": int(len(group)),
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
                values.std(
                    ddof=0
                )
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        [
            "macro_f1_mean",
            "balanced_accuracy_mean",
            "roc_auc_macro_ovr_mean",
            "eer_macro_ovr_diagnostic_mean",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    return summary


def save_best_diagnostics(
    predictions_df: pd.DataFrame,
    best_row: pd.Series,
    users: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    best_predictions = predictions_df[
        (
            predictions_df["model"]
            .astype(str)
            .eq(
                str(
                    best_row["model"]
                )
            )
        )
        & (
            predictions_df["window_size"]
            == int(
                best_row["window_size"]
            )
        )
        & (
            predictions_df["overlap_pct"]
            == int(
                best_row["overlap_pct"]
            )
        )
    ].copy()

    if best_predictions.empty:
        raise RuntimeError(
            "Predições OOF do melhor modelo não encontradas."
        )

    matrix = confusion_matrix(
        best_predictions[
            "y_true_user"
        ],
        best_predictions[
            "y_pred_user"
        ],
        labels=users,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"real__{u}"
            for u in users
        ],
        columns=[
            f"pred__{u}"
            for u in users
        ],
    )

    report = classification_report(
        best_predictions[
            "y_true_user"
        ],
        best_predictions[
            "y_pred_user"
        ],
        labels=users,
        output_dict=True,
        zero_division=0,
    )

    per_user_rows = []

    for user in users:
        row = report[user]

        per_user_rows.append(
            {
                "user": user,
                "precision": float(
                    row["precision"]
                ),
                "recall": float(
                    row["recall"]
                ),
                "f1": float(
                    row["f1-score"]
                ),
                "support": int(
                    row["support"]
                ),
            }
        )

    per_user_df = (
        pd.DataFrame(
            per_user_rows
        )
        .sort_values(
            "f1",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    matrix_df.to_csv(
        CONFUSION_PATH,
        encoding="utf-8-sig",
    )

    per_user_df.to_csv(
        USER_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return matrix_df, per_user_df


def train_final_model(
    df: pd.DataFrame,
    best_row: pd.Series,
    features: list[str],
    label_encoder: LabelEncoder,
) -> Pipeline:

    subset = df[
        (
            df["window_size"]
            == int(
                best_row["window_size"]
            )
        )
        & (
            df["requested_overlap_pct"]
            == int(
                best_row["overlap_pct"]
            )
        )
    ].copy()

    X = subset[
        features
    ]

    y = label_encoder.transform(
        subset[
            TARGET_COLUMN
        ].astype(str)
    )

    model_name = str(
        best_row["model"]
    )

    if model_name == "RandomForest":
        model = build_random_forest()
        model.fit(
            X,
            y,
        )

    elif model_name == "XGBoost":
        model = build_xgboost(
            n_classes=len(
                label_encoder.classes_
            )
        )

        sample_weight = (
            compute_sample_weight(
                class_weight="balanced",
                y=y,
            )
        )

        model.fit(
            X,
            y,
            model__sample_weight=sample_weight,
        )

    else:
        raise RuntimeError(
            f"Modelo desconhecido: {model_name}"
        )

    return model


def main() -> None:
    print("=" * 100)
    print(
        "CYBELL-J — EXPERIMENTO 2 — "
        "RECONHECIMENTO GLOBAL DE 6 USUÁRIAS USANDO TECLADO"
    )
    print("=" * 100)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado:\n{DATASET_PATH}"
        )

    print("\n1) Carregando dataset...")

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
    }

    missing = sorted(
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Colunas obrigatórias ausentes: "
            f"{missing}"
        )

    features = load_feature_contract()

    validate_features(
        df,
        features,
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
            f"Esperadas 6 usuárias, encontradas {len(users)}: {users}"
        )

    print(
        f"   Usuárias: {len(users)}"
    )
    print(
        "   " + ", ".join(users)
    )
    print(
        f"   Features comportamentais: {len(features)}"
    )

    print("\n2) Sessões disponíveis por usuária:")

    session_counts = (
        df.groupby(
            TARGET_COLUMN
        )["session_id"]
        .nunique()
        .sort_index()
    )

    print(
        session_counts.to_string()
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(users)

    all_fold_results = []
    all_predictions = []
    all_eer_rows = []

    total_configs = (
        len(WINDOW_SIZES)
        * len(OVERLAPS)
    )

    config_number = 0

    print("\n" + "=" * 100)
    print("3) TREINANDO AS 12 CONFIGURAÇÕES")
    print("=" * 100)

    for window_size in WINDOW_SIZES:
        for overlap_pct in OVERLAPS:
            config_number += 1

            config = df[
                (
                    df["window_size"]
                    == window_size
                )
                & (
                    df["requested_overlap_pct"]
                    == overlap_pct
                )
            ].copy()

            print(
                f"\n[{config_number}/{total_configs}] "
                f"{window_size} eventos | "
                f"overlap {overlap_pct}%"
            )
            print(
                f"   Janelas: "
                f"{len(config):,}".replace(
                    ",",
                    ".",
                )
            )

            if config.empty:
                print(
                    "   AVISO: sem dados. Pulando."
                )
                continue

            config_users = set(
                config[
                    TARGET_COLUMN
                ]
                .astype(str)
                .unique()
            )

            if config_users != set(users):
                print(
                    "   AVISO: nem todas as usuárias estão presentes. Pulando."
                )
                continue

            try:
                folds = build_session_folds(
                    config,
                    n_folds=N_FOLDS,
                )
            except RuntimeError as exc:
                print(
                    f"   AVISO: {exc}"
                )
                print(
                    "   Configuração pulada."
                )
                continue

            for fold_info in folds:
                fold_number = int(
                    fold_info["fold"]
                )

                train_sessions = (
                    fold_info["train_sessions"]
                )

                test_sessions = (
                    fold_info["test_sessions"]
                )

                train = config[
                    config["session_id"]
                    .astype(str)
                    .isin(train_sessions)
                ].copy()

                test = config[
                    config["session_id"]
                    .astype(str)
                    .isin(test_sessions)
                ].copy()

                models = {
                    "RandomForest": build_random_forest(),
                    "XGBoost": build_xgboost(
                        n_classes=len(users)
                    ),
                }

                print(
                    f"   Fold {fold_number}: "
                    f"treino={len(train):,} | "
                    f"teste={len(test):,}".replace(
                        ",",
                        ".",
                    )
                )

                for model_name, model in models.items():
                    (
                        result,
                        pred_df,
                        eer_rows,
                    ) = train_one_fold(
                        model_name=model_name,
                        model=model,
                        train=train,
                        test=test,
                        features=features,
                        label_encoder=label_encoder,
                        window_size=window_size,
                        overlap_pct=overlap_pct,
                        fold_number=fold_number,
                    )

                    all_fold_results.append(
                        result
                    )

                    all_predictions.append(
                        pred_df
                    )

                    all_eer_rows.extend(
                        eer_rows
                    )

                    print(
                        f"      {model_name:<13} "
                        f"MacroF1={result['macro_f1']:.4f} | "
                        f"BalAcc={result['balanced_accuracy']:.4f} | "
                        f"AUC={result['roc_auc_macro_ovr']:.4f} | "
                        f"EERovr={result['eer_macro_ovr_diagnostic']:.4f}"
                    )

    if not all_fold_results:
        raise RuntimeError(
            "Nenhum treinamento foi concluído."
        )

    folds_df = pd.DataFrame(
        all_fold_results
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    eer_users_df = pd.DataFrame(
        all_eer_rows
    )

    folds_df.to_csv(
        FOLDS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    eer_users_df.to_csv(
        EER_USERS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary = summarize_results(
        folds_df
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("4) TOP 10 CONFIGURAÇÕES")
    print("=" * 100)

    display_columns = [
        "rank",
        "model",
        "window_size",
        "overlap_pct",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "accuracy_mean",
        "roc_auc_macro_ovr_mean",
        "eer_macro_ovr_diagnostic_mean",
    ]

    print(
        summary[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    best = summary.iloc[0]

    print("\n" + "=" * 100)
    print("MELHOR MODELO GLOBAL DE TECLADO")
    print("=" * 100)

    print(
        f"\nModelo: {best['model']}"
    )
    print(
        f"Janela: {int(best['window_size'])} eventos"
    )
    print(
        f"Overlap: {int(best['overlap_pct'])}%"
    )
    print(
        f"Macro F1 médio: {best['macro_f1_mean']:.4f}"
    )
    print(
        f"Balanced Accuracy média: "
        f"{best['balanced_accuracy_mean']:.4f}"
    )
    print(
        f"Accuracy média: "
        f"{best['accuracy_mean']:.4f}"
    )
    print(
        f"ROC-AUC macro OVR: "
        f"{best['roc_auc_macro_ovr_mean']:.4f}"
    )
    print(
        f"EER macro OVR diagnóstico: "
        f"{best['eer_macro_ovr_diagnostic_mean']:.4f}"
    )

    matrix_df, per_user_df = save_best_diagnostics(
        predictions_df=predictions_df,
        best_row=best,
        users=users,
    )

    print(
        "\n5) F1 por usuária no melhor modelo:"
    )
    print(
        per_user_df.to_string(
            index=False
        )
    )

    print(
        "\n6) Matriz de confusão OOF:"
    )
    print(
        matrix_df.to_string()
    )

    print(
        "\n7) Treinando o melhor modelo global "
        "com todas as sessões..."
    )

    final_model = train_final_model(
        df=df,
        best_row=best,
        features=features,
        label_encoder=label_encoder,
    )

    bundle = {
        "model": final_model,
        "label_encoder": label_encoder,
        "features": features,
        "window_size": int(
            best["window_size"]
        ),
        "requested_overlap_pct": int(
            best["overlap_pct"]
        ),
        "algorithm": str(
            best["model"]
        ),
        "task": "keyboard_global_multiclass",
    }

    joblib.dump(
        bundle,
        MODEL_PATH,
    )

    feature_info = {
        "features": features,
        "n_features": len(features),
        "window_size": int(
            best["window_size"]
        ),
        "requested_overlap_pct": int(
            best["overlap_pct"]
        ),
        "algorithm": str(
            best["model"]
        ),
        "classes": users,
    }

    with FEATURES_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_info,
            f,
            indent=4,
            ensure_ascii=False,
        )

    manifesto = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": (
            "Experimento 2 - teclado global"
        ),
        "task": (
            "classificacao multiclasses de 6 usuarias"
        ),
        "dataset": str(
            DATASET_PATH.relative_to(
                BASE_DIR
            )
        ),
        "protocol": {
            "window_sizes": WINDOW_SIZES,
            "overlaps_pct": OVERLAPS,
            "n_folds": N_FOLDS,
            "split": "session_id",
            "imputation": "median",
            "random_state": RANDOM_STATE,
            "algorithms": [
                "RandomForest",
                "XGBoost",
            ],
            "selection_metric": "macro_f1",
        },
        "best": {
            "algorithm": str(
                best["model"]
            ),
            "window_size": int(
                best["window_size"]
            ),
            "overlap_pct": int(
                best["overlap_pct"]
            ),
            "macro_f1_mean": float(
                best["macro_f1_mean"]
            ),
            "balanced_accuracy_mean": float(
                best["balanced_accuracy_mean"]
            ),
            "accuracy_mean": float(
                best["accuracy_mean"]
            ),
            "roc_auc_macro_ovr_mean": float(
                best["roc_auc_macro_ovr_mean"]
            ),
            "eer_macro_ovr_diagnostic_mean": float(
                best[
                    "eer_macro_ovr_diagnostic_mean"
                ]
            ),
        },
        "n_features": len(features),
        "users": users,
        "next_step": (
            "Executar o Experimento 1 individualizado "
            "somente com os 2 indivíduos definidos."
        ),
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifesto,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print("EXPERIMENTO 2 — TECLADO GLOBAL CONCLUÍDO")
    print("=" * 100)

    print("\nArquivos gerados:")
    print(f"  {FOLDS_PATH}")
    print(f"  {SUMMARY_PATH}")
    print(f"  {PREDICTIONS_PATH}")
    print(f"  {EER_USERS_PATH}")
    print(f"  {CONFUSION_PATH}")
    print(f"  {USER_METRICS_PATH}")
    print(f"  {MODEL_PATH}")
    print(f"  {FEATURES_PATH}")
    print(f"  {MANIFEST_PATH}")

    print(
        "\nPróxima etapa: Experimento 1 individualizado "
        "com somente 2 indivíduos."
    )


if __name__ == "__main__":
    main()
