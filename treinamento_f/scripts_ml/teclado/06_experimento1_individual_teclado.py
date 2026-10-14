"""
06_experimento1_individual_teclado.py

CYBELL-J — Experimento 1
AUTENTICAÇÃO INDIVIDUALIZADA USANDO APENAS TECLADO

Prova de conceito com SOMENTE 2 indivíduos:
- Yass -> Random Forest
- Lina -> XGBoost

Cada modelo é binário 1 x resto:
- classe 1: usuária genuína
- classe 0: demais usuárias como impostoras

Protocolo:
- usa a melhor configuração de janela/overlap encontrada no Experimento 2;
- 3 folds por session_id;
- nenhuma sessão aparece em treino e teste ao mesmo tempo;
- SimpleImputer(strategy="median");
- impostores de treino limitados a 2:1 em relação às janelas genuínas;
- teste com todos os dados das sessões de teste;
- métricas: EER, AUC, F1, balanced accuracy, accuracy, precision e recall.
"""

from __future__ import annotations

import hashlib
import json
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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "data" / "teclado_dataset" / "dataset_teclado.csv"
DATASET_MANIFEST_PATH = BASE_DIR / "data" / "teclado_dataset" / "dataset_teclado_manifest.json"
EXP2_MANIFEST_PATH = BASE_DIR / "data" / "teclado_experimento2" / "manifesto_experimento2.json"

OUT_DIR = BASE_DIR / "data" / "teclado_experimento1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUT_DIR / "resultados_folds.csv"
SUMMARY_PATH = OUT_DIR / "resumo_individuos.csv"
PREDICTIONS_PATH = OUT_DIR / "predicoes_oof.csv"
THRESHOLDS_PATH = OUT_DIR / "thresholds_eer.json"
MANIFEST_PATH = OUT_DIR / "manifesto_experimento1.json"

RANDOM_STATE = 42
N_FOLDS = 3
IMPOSTOR_RATIO = 2.0

INDIVIDUALS = {
    "yass": "RandomForest",
    "lina": "XGBoost",
}


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (RANDOM_STATE + int(digest[:8], 16)) % (2**32 - 1)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_contract() -> tuple[list[str], int, int]:
    dataset_manifest = load_json(DATASET_MANIFEST_PATH)
    exp2_manifest = load_json(EXP2_MANIFEST_PATH)

    features = dataset_manifest.get("feature_columns", [])
    if not features:
        raise RuntimeError("feature_columns ausente no manifesto do dataset.")

    best = exp2_manifest.get("best", {})
    return features, int(best["window_size"]), int(best["overlap_pct"])


def build_session_folds(df: pd.DataFrame) -> list[dict]:
    users = sorted(df["user"].astype(str).unique().tolist())
    fold_test_sessions = [[] for _ in range(N_FOLDS)]

    for user in users:
        sessions = sorted(
            df.loc[df["user"].astype(str).eq(user), "session_id"]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(sessions) < N_FOLDS:
            raise RuntimeError(
                f"{user}: apenas {len(sessions)} sessões na configuração escolhida."
            )

        rng = np.random.default_rng(stable_seed(f"folds|{user}"))
        rng.shuffle(sessions)

        for idx, session in enumerate(sessions):
            fold_test_sessions[idx % N_FOLDS].append(session)

    all_sessions = set(df["session_id"].astype(str).unique().tolist())
    folds = []

    for fold_idx in range(N_FOLDS):
        test_sessions = set(fold_test_sessions[fold_idx])
        train_sessions = all_sessions - test_sessions

        if train_sessions & test_sessions:
            raise RuntimeError("Vazamento de session_id.")

        folds.append(
            {
                "fold": fold_idx + 1,
                "train_sessions": sorted(train_sessions),
                "test_sessions": sorted(test_sessions),
            }
        )

    return folds


def sample_impostors(
    df: pd.DataFrame,
    impostor_idx: np.ndarray,
    n_genuine: int,
    protected_user: str,
    fold: int,
) -> np.ndarray:
    max_impostors = int(round(n_genuine * IMPOSTOR_RATIO))

    if len(impostor_idx) <= max_impostors:
        return impostor_idx

    pool = df.loc[impostor_idx, ["user"]].copy()
    pool["_idx"] = impostor_idx

    users = sorted(pool["user"].astype(str).unique().tolist())
    selected = []
    remaining = max_impostors

    for pos, user in enumerate(users):
        idx_user = pool.loc[
            pool["user"].astype(str).eq(user), "_idx"
        ].to_numpy(dtype=np.int64)

        groups_left = len(users) - pos
        take = min(len(idx_user), max(1, remaining // groups_left))

        rng = np.random.default_rng(
            stable_seed(f"imp|{protected_user}|fold{fold}|{user}")
        )
        selected.append(rng.choice(idx_user, size=take, replace=False))
        remaining -= take

        if remaining <= 0:
            break

    out = np.concatenate(selected).astype(np.int64)

    if len(out) > max_impostors:
        rng = np.random.default_rng(
            stable_seed(f"trim|{protected_user}|fold{fold}")
        )
        out = rng.choice(out, size=max_impostors, replace=False)

    return out


def build_model(
    algorithm: str,
    protected_user: str,
    fold: int,
) -> Pipeline:
    if algorithm == "RandomForest":
        classifier = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=stable_seed(f"rf|{protected_user}|{fold}"),
            n_jobs=-1,
        )
    elif algorithm == "XGBoost":
        classifier = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=stable_seed(f"xgb|{protected_user}|{fold}"),
            n_jobs=-1,
            verbosity=0,
        )
    else:
        raise RuntimeError(f"Algoritmo desconhecido: {algorithm}")

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", classifier),
        ]
    )


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def calculate_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    eer, threshold = compute_eer(y_true, scores)
    pred = (scores >= threshold).astype(int)

    return {
        "eer": eer,
        "threshold_eer": threshold,
        "auc": float(roc_auc_score(y_true, scores)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def fit_one_fold(
    df: pd.DataFrame,
    features: list[str],
    protected_user: str,
    algorithm: str,
    fold_info: dict,
) -> tuple[dict, pd.DataFrame]:
    fold = int(fold_info["fold"])

    train_mask = df["session_id"].astype(str).isin(fold_info["train_sessions"])
    test_mask = df["session_id"].astype(str).isin(fold_info["test_sessions"])

    genuine_train_idx = np.flatnonzero(
        (train_mask & df["user"].astype(str).eq(protected_user)).to_numpy()
    ).astype(np.int64)

    impostor_pool_idx = np.flatnonzero(
        (train_mask & ~df["user"].astype(str).eq(protected_user)).to_numpy()
    ).astype(np.int64)

    test_idx = np.flatnonzero(test_mask.to_numpy()).astype(np.int64)

    impostor_train_idx = sample_impostors(
        df,
        impostor_pool_idx,
        len(genuine_train_idx),
        protected_user,
        fold,
    )

    train_idx = np.concatenate([genuine_train_idx, impostor_train_idx])

    rng = np.random.default_rng(
        stable_seed(f"shuffle|{protected_user}|fold{fold}")
    )
    rng.shuffle(train_idx)

    X_train = df.loc[train_idx, features].copy()
    X_test = df.loc[test_idx, features].copy()

    y_train = (
        df.loc[train_idx, "user"]
        .astype(str)
        .eq(protected_user)
        .astype(int)
        .to_numpy()
    )

    y_test = (
        df.loc[test_idx, "user"]
        .astype(str)
        .eq(protected_user)
        .astype(int)
        .to_numpy()
    )

    if len(np.unique(y_test)) < 2:
        raise RuntimeError(
            f"{protected_user} fold {fold}: teste sem as duas classes."
        )

    model = build_model(algorithm, protected_user, fold)

    fit_kwargs = {}

    if algorithm == "XGBoost":
        positive = int(np.sum(y_train == 1))
        negative = int(np.sum(y_train == 0))

        weights = np.ones(len(y_train), dtype=float)
        if positive > 0:
            weights[y_train == 1] = negative / positive

        fit_kwargs["model__sample_weight"] = weights

    model.fit(X_train, y_train, **fit_kwargs)
    scores = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, scores)

    result = {
        "protected_user": protected_user,
        "algorithm": algorithm,
        "fold": fold,
        "n_genuine_train": int(len(genuine_train_idx)),
        "n_impostor_train": int(len(impostor_train_idx)),
        "n_test": int(len(test_idx)),
        "n_genuine_test": int(np.sum(y_test == 1)),
        "n_impostor_test": int(np.sum(y_test == 0)),
        **metrics,
    }

    pred_df = pd.DataFrame(
        {
            "protected_user": protected_user,
            "algorithm": algorithm,
            "fold": fold,
            "session_id": df.loc[test_idx, "session_id"].astype(str).to_numpy(),
            "window_id": df.loc[test_idx, "window_id"].astype(str).to_numpy(),
            "actual_user": df.loc[test_idx, "user"].astype(str).to_numpy(),
            "y_true_genuine": y_test,
            "score_genuine": scores,
        }
    )

    return result, pred_df


def train_final_bundle(
    df: pd.DataFrame,
    features: list[str],
    protected_user: str,
    algorithm: str,
    threshold: float,
    window_size: int,
    overlap_pct: int,
) -> Path:
    genuine_idx = np.flatnonzero(
        df["user"].astype(str).eq(protected_user).to_numpy()
    ).astype(np.int64)

    impostor_pool_idx = np.flatnonzero(
        (~df["user"].astype(str).eq(protected_user)).to_numpy()
    ).astype(np.int64)

    impostor_idx = sample_impostors(
        df,
        impostor_pool_idx,
        len(genuine_idx),
        protected_user,
        999,
    )

    train_idx = np.concatenate([genuine_idx, impostor_idx])

    rng = np.random.default_rng(stable_seed(f"final|{protected_user}"))
    rng.shuffle(train_idx)

    X = df.loc[train_idx, features].copy()
    y = (
        df.loc[train_idx, "user"]
        .astype(str)
        .eq(protected_user)
        .astype(int)
        .to_numpy()
    )

    model = build_model(algorithm, protected_user, 999)

    fit_kwargs = {}
    if algorithm == "XGBoost":
        positive = int(np.sum(y == 1))
        negative = int(np.sum(y == 0))
        weights = np.ones(len(y), dtype=float)
        if positive > 0:
            weights[y == 1] = negative / positive
        fit_kwargs["model__sample_weight"] = weights

    model.fit(X, y, **fit_kwargs)

    bundle = {
        "model": model,
        "protected_user": protected_user,
        "algorithm": algorithm,
        "features": features,
        "window_size": window_size,
        "requested_overlap_pct": overlap_pct,
        "threshold_eer_oof": threshold,
        "task": "keyboard_authentication_1to1",
    }

    output = OUT_DIR / f"{protected_user}_{algorithm.lower()}.joblib"
    joblib.dump(bundle, output)
    return output


def main() -> None:
    print("=" * 100)
    print(
        "CYBELL-J — EXPERIMENTO 1 — "
        "AUTENTICAÇÃO INDIVIDUALIZADA DE TECLADO"
    )
    print("=" * 100)

    print("\nIndivíduos da prova de conceito:")
    print("  Yass -> Random Forest")
    print("  Lina -> XGBoost")
    print("\nSomente estes 2 modelos individuais serão treinados.")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    features, window_size, overlap_pct = load_contract()

    print("\nConfiguração herdada do Experimento 2:")
    print(f"  janela = {window_size} eventos")
    print(f"  overlap = {overlap_pct}%")
    print(f"  features = {len(features)}")

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise RuntimeError(
            "Features ausentes: " + ", ".join(missing_features)
        )

    df = df[
        (df["window_size"] == window_size)
        & (df["requested_overlap_pct"] == overlap_pct)
    ].copy().reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            "Nenhuma janela encontrada para a configuração vencedora."
        )

    print(f"\nJanelas utilizadas: {len(df):,}".replace(",", "."))

    base_table = (
        df.groupby("user")
        .agg(
            windows=("window_id", "size"),
            sessions=("session_id", "nunique"),
        )
        .reset_index()
        .sort_values("user")
    )

    print("\nBase da configuração:")
    print(base_table.to_string(index=False))

    folds = build_session_folds(df)

    results = []
    predictions = []

    for protected_user, algorithm in INDIVIDUALS.items():
        print("\n" + "=" * 100)
        print(f"{protected_user.upper()} — {algorithm}")
        print("=" * 100)

        for fold_info in folds:
            result, pred_df = fit_one_fold(
                df,
                features,
                protected_user,
                algorithm,
                fold_info,
            )
            results.append(result)
            predictions.append(pred_df)

            print(
                f"Fold {result['fold']}: "
                f"EER={result['eer']:.4f} | "
                f"AUC={result['auc']:.4f} | "
                f"F1={result['f1']:.4f} | "
                f"BalAcc={result['balanced_accuracy']:.4f}"
            )

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(predictions, ignore_index=True)

    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(
        PREDICTIONS_PATH, index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    thresholds = {}
    model_paths = {}

    for (protected_user, algorithm), group in results_df.groupby(
        ["protected_user", "algorithm"], sort=False
    ):
        user_predictions = predictions_df[
            predictions_df["protected_user"].astype(str).eq(protected_user)
        ]

        global_oof = calculate_metrics(
            user_predictions["y_true_genuine"].to_numpy(dtype=int),
            user_predictions["score_genuine"].to_numpy(dtype=float),
        )

        row = {
            "protected_user": protected_user,
            "algorithm": algorithm,
            "n_folds": int(len(group)),
            "eer_mean_folds": float(group["eer"].mean()),
            "eer_std_folds": float(group["eer"].std(ddof=0)),
            "auc_mean_folds": float(group["auc"].mean()),
            "f1_mean_folds": float(group["f1"].mean()),
            "balanced_accuracy_mean_folds": float(
                group["balanced_accuracy"].mean()
            ),
            "eer_oof": float(global_oof["eer"]),
            "threshold_oof": float(global_oof["threshold_eer"]),
            "auc_oof": float(global_oof["auc"]),
            "f1_oof": float(global_oof["f1"]),
            "balanced_accuracy_oof": float(
                global_oof["balanced_accuracy"]
            ),
        }

        summary_rows.append(row)

        thresholds[protected_user] = {
            "algorithm": algorithm,
            "threshold_eer_oof": float(global_oof["threshold_eer"]),
            "eer_oof": float(global_oof["eer"]),
        }

        model_path = train_final_bundle(
            df,
            features,
            protected_user,
            algorithm,
            float(global_oof["threshold_eer"]),
            window_size,
            overlap_pct,
        )
        model_paths[protected_user] = str(
            model_path.relative_to(BASE_DIR)
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    with THRESHOLDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=4, ensure_ascii=False)

    manifesto = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Experimento 1 - teclado individualizado",
        "individuals": INDIVIDUALS,
        "protocol": {
            "task": "binary one-vs-rest",
            "window_size": window_size,
            "overlap_pct": overlap_pct,
            "n_folds": N_FOLDS,
            "split": "session_id",
            "imputation": "median",
            "impostor_train_ratio_max": IMPOSTOR_RATIO,
            "test_uses_all_available_windows": True,
            "random_state": RANDOM_STATE,
        },
        "n_features": len(features),
        "models": model_paths,
        "results": summary_rows,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifesto, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("RESULTADO FINAL — EXPERIMENTO 1")
    print("=" * 100)

    print(
        summary_df[
            [
                "protected_user",
                "algorithm",
                "eer_oof",
                "auc_oof",
                "f1_oof",
                "balanced_accuracy_oof",
                "threshold_oof",
            ]
        ].to_string(index=False)
    )

    print("\nModelos salvos:")
    for user, path in model_paths.items():
        print(f"  {user}: {path}")

    print("\nEXPERIMENTO 1 CONCLUÍDO COM SOMENTE 2 INDIVÍDUOS.")


if __name__ == "__main__":
    main()
