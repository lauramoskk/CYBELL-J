"""
11B_controle_quantidade_perfil_vs_subperfil.py

CYBELL-J — Experimento 11B
Controle por quantidade de dados:
PERFIL_UNICO_CONTROLADO x SUBPERFIL_MOUSE

- Usuária protegida: Bianca
- Configuração: 50 eventos / 75% overlap
- Teste: somente mouse
- 3 folds externos por session_id
- 5 repetições pareadas
- Mesmo número de genuínos nas duas estratégias
- Mesmos impostores nas duas estratégias
- Mesmo teste nas duas estratégias
- PyCaret continua presente: LR / RF / ExtraTrees
- XGBoost usa GPU apenas em bursts leves
- CPU limitada a 2 threads
- float32 + leitura em chunks + gravação incremental
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
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

try:
    import pycaret
    try:
        from pycaret.tasks import ClassificationExperiment
    except ImportError:
        from pycaret.classification import ClassificationExperiment
except ImportError as exc:
    raise SystemExit(
        "PyCaret não encontrado. Ative .venv_pycaret."
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "XGBoost não encontrado. Instale com: python -m pip install xgboost"
    ) from exc


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "mouse_dataset" / "dataset_device_high_confidence.csv"
OUT = BASE_DIR / "data" / "profile_vs_mouse_subprofile_controlled"
LB_DIR = OUT / "leaderboards_pycaret"
OUT.mkdir(parents=True, exist_ok=True)
LB_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUT / "resultados_folds_repeticoes.csv"
SUMMARY_PATH = OUT / "resumo_estrategias_modelos.csv"
WINNERS_PATH = OUT / "vencedores_por_estrategia.csv"
COMPARISON_PATH = OUT / "comparacao_final.csv"
PAIRED_PATH = OUT / "comparacao_pareada.csv"
COMPOSITION_PATH = OUT / "composicao_amostras.csv"
PRED_PATH = OUT / "predicoes_oof.csv"
MANIFEST_PATH = OUT / "manifesto_11b.json"

PROTECTED_USER = "bianca"
WINDOW_SIZE = 50
OVERLAP = 75
N_OUTER_FOLDS = 3
N_INNER_FOLDS = 2
N_REPEATS = 5
RANDOM_STATE = 42

CPU_JOBS = 2
PYCARET_MODELS = ["lr", "rf", "et"]

IMPOSTOR_RATIO = 2.0

# XGBoost leve para não castigar a GPU
XGB_TREES = 100
XGB_DEPTH = 4
XGB_MAX_BIN = 128

CHUNK = 50_000

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_JOBS))

META = {
    "target_user","user","session_id","window_id","window_size",
    "requested_overlap_pct","step_events","effective_overlap_pct",
    "window_start_ms","window_end_ms","n_events",
    "device_mouse_pct","device_trackpad_pct","device_context",
    "device_confidence","device_high_confidence","device_label_source",
    "target_user_device",
}


def seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (RANDOM_STATE + int(h[:8], 16)) % (2**32 - 1)


def reset_outputs():
    for p in [RESULTS_PATH, COMPOSITION_PATH, PRED_PATH]:
        if p.exists():
            p.unlink()


def append_csv(df: pd.DataFrame, path: Path):
    if df.empty:
        return
    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def load_data() -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK, low_memory=False):
        part = chunk[
            (chunk["window_size"] == WINDOW_SIZE)
            & (chunk["requested_overlap_pct"] == OVERLAP)
            & chunk["device_context"].isin(["mouse", "trackpad"])
        ].copy()
        if not part.empty:
            parts.append(part)
        del chunk, part
        gc.collect()

    if not parts:
        raise RuntimeError("Nenhum dado 50/75 encontrado.")

    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        c for c in df.columns
        if c not in META and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        raise RuntimeError("Nenhuma feature numérica encontrada.")
    return cols


def compact(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def build_outer_folds(mouse_df: pd.DataFrame) -> list[dict]:
    users = sorted(mouse_df["user"].astype(str).unique())
    fold_tests = [[] for _ in range(N_OUTER_FOLDS)]

    for user in users:
        sessions = sorted(
            mouse_df.loc[
                mouse_df["user"].astype(str).eq(user),
                "session_id",
            ].astype(str).unique().tolist()
        )
        if len(sessions) < N_OUTER_FOLDS:
            raise RuntimeError(f"{user} possui apenas {len(sessions)} sessões mouse.")

        rng = np.random.default_rng(seed(f"outer|{user}"))
        rng.shuffle(sessions)

        for i, s in enumerate(sessions):
            fold_tests[i % N_OUTER_FOLDS].append(s)

    return [
        {"fold": i + 1, "test_sessions": sorted(set(fold_tests[i]))}
        for i in range(N_OUTER_FOLDS)
    ]


def fold_pools(all_df: pd.DataFrame, mouse_df: pd.DataFrame, fold: dict) -> dict:
    test_sessions = set(fold["test_sessions"])

    test = mouse_df[
        mouse_df["session_id"].astype(str).isin(test_sessions)
    ].copy()

    genuine_mouse = mouse_df[
        mouse_df["user"].astype(str).eq(PROTECTED_USER)
        & ~mouse_df["session_id"].astype(str).isin(test_sessions)
    ].copy()

    genuine_mixed = all_df[
        all_df["user"].astype(str).eq(PROTECTED_USER)
        & ~all_df["session_id"].astype(str).isin(test_sessions)
    ].copy()

    impostors = mouse_df[
        ~mouse_df["user"].astype(str).eq(PROTECTED_USER)
        & ~mouse_df["session_id"].astype(str).isin(test_sessions)
    ].copy()

    train_sessions = set(
        pd.concat([
            genuine_mixed[["session_id"]],
            impostors[["session_id"]],
        ])["session_id"].astype(str)
    )
    test_s = set(test["session_id"].astype(str))
    if train_sessions & test_s:
        raise RuntimeError("Vazamento de session_id.")

    return {
        "test": test,
        "genuine_mouse": genuine_mouse,
        "genuine_mixed": genuine_mixed,
        "impostors": impostors,
    }


def sample_impostors(pool: pd.DataFrame, n: int, fold: int, repeat: int) -> pd.DataFrame:
    users = sorted(pool["user"].astype(str).unique())
    parts = []
    remaining = n

    for i, user in enumerate(users):
        p = pool[pool["user"].astype(str).eq(user)]
        left = len(users) - i
        take = min(len(p), max(1, remaining // left))
        if take:
            parts.append(
                p.sample(
                    n=take,
                    random_state=seed(f"imp|{fold}|{repeat}|{user}"),
                    replace=False,
                )
            )
            remaining -= take

    return pd.concat(parts, ignore_index=False)


def paired_train_sets(pools: dict, fold: int, repeat: int):
    gm = pools["genuine_mouse"]
    mixed_pool = pools["genuine_mixed"]
    n_genuine = len(gm)
    n_imp = int(round(n_genuine * IMPOSTOR_RATIO))

    imp = sample_impostors(pools["impostors"], n_imp, fold, repeat)

    mixed = mixed_pool.sample(
        n=n_genuine,
        replace=False,
        random_state=seed(f"mixed|{fold}|{repeat}"),
    )

    unique_train = pd.concat([mixed, imp], ignore_index=False).sample(
        frac=1.0,
        random_state=seed(f"shuffle_unique|{fold}|{repeat}"),
    )
    sub_train = pd.concat([gm, imp], ignore_index=False).sample(
        frac=1.0,
        random_state=seed(f"shuffle_sub|{fold}|{repeat}"),
    )

    composition = {
        "fold": fold,
        "repeat": repeat,
        "n_genuine_each_strategy": n_genuine,
        "n_impostor_each_strategy": len(imp),
        "mixed_genuine_mouse": int(
            mixed["device_context"].astype(str).eq("mouse").sum()
        ),
        "mixed_genuine_trackpad": int(
            mixed["device_context"].astype(str).eq("trackpad").sum()
        ),
        "subprofile_genuine_mouse": n_genuine,
        "subprofile_genuine_trackpad": 0,
    }

    return unique_train, sub_train, composition


class SessionCV(BaseCrossValidator):
    def __init__(self, index_to_session: dict, fold_defs: list[dict]):
        self.index_to_session = index_to_session
        self.fold_defs = fold_defs

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.fold_defs)

    def split(self, X, y=None, groups=None):
        sessions = np.array(
            [self.index_to_session.get(i) for i in X.index],
            dtype=object,
        )

        for fd in self.fold_defs:
            test_sessions = set(fd["test_sessions"])
            mask = np.array([s in test_sessions for s in sessions], dtype=bool)
            tr = np.flatnonzero(~mask)
            te = np.flatnonzero(mask)

            if len(tr) == 0 or len(te) == 0:
                continue
            if y is not None:
                yy = np.asarray(y)
                if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[te])) < 2:
                    continue

            yield tr, te


def inner_folds(train_df: pd.DataFrame) -> list[dict]:
    tmp = train_df[["session_id", "user"]].drop_duplicates().copy()
    tmp["target_bin"] = tmp["user"].astype(str).eq(PROTECTED_USER).astype(int)

    buckets = [[] for _ in range(N_INNER_FOLDS)]

    for label in [0, 1]:
        sessions = sorted(
            tmp.loc[tmp["target_bin"].eq(label), "session_id"]
            .astype(str).unique().tolist()
        )
        if len(sessions) < N_INNER_FOLDS:
            raise RuntimeError(
                f"Poucas sessões no CV interno para classe {label}: {len(sessions)}"
            )

        rng = np.random.default_rng(seed(f"inner|{label}|{len(train_df)}"))
        rng.shuffle(sessions)

        for i, s in enumerate(sessions):
            buckets[i % N_INNER_FOLDS].append(s)

    return [
        {"test_sessions": sorted(set(b))}
        for b in buckets
    ]


def pycaret_select(train_df: pd.DataFrame, features: list[str], strategy: str, fold: int):
    work = train_df[features + ["user", "session_id"]].copy()
    work["target_bin"] = work["user"].astype(str).eq(PROTECTED_USER).astype(int)

    cv = SessionCV(
        index_to_session=work["session_id"].astype(str).to_dict(),
        fold_defs=inner_folds(work),
    )

    py_data = work[features + ["target_bin"]].copy()

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
        use_gpu=False,
        verbose=False,
    )
    exp.fit(py_data)

    result = exp.compare_models(
        fold=cv,
        include=PYCARET_MODELS,
        sort="AUC",
        n_select=1,
        turbo=True,
        errors="ignore",
        round=4,
        verbose=False,
    )

    result.leaderboard.to_csv(
        LB_DIR / f"{strategy}_fold{fold}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not result.models:
        raise RuntimeError("PyCaret não selecionou modelo.")

    pipeline = result.models[0]
    model_id = result.ranked_ids[0] if result.ranked_ids else "best"

    del exp, result, py_data, work
    gc.collect()

    return pipeline, f"PyCaret_{model_id}"


def gpu_available() -> bool:
    try:
        rng = np.random.default_rng(RANDOM_STATE)
        X = rng.random((128, 8), dtype=np.float32)
        y = rng.integers(0, 2, 128)
        m = XGBClassifier(
            n_estimators=2,
            max_depth=2,
            max_bin=64,
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            verbosity=0,
        )
        m.fit(X, y)
        del X, y, m
        gc.collect()
        return True
    except Exception:
        return False


def build_xgb(y: np.ndarray, use_gpu: bool) -> Pipeline:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    spw = neg / pos if pos else 1.0

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=XGB_TREES,
            learning_rate=0.07,
            max_depth=XGB_DEPTH,
            min_child_weight=2,
            subsample=0.80,
            colsample_bytree=0.80,
            max_bin=XGB_MAX_BIN,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=CPU_JOBS,
            verbosity=0,
        )),
    ])


def score_genuine(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        classes = list(getattr(model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else 1
        return p[:, idx].astype("float32")
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype="float32")
    return np.asarray(model.predict(X), dtype="float32")


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))

    far = float(fpr[idx])
    frr = float(fnr[idx])
    eer = float((far + frr) / 2.0)
    thr = float(thresholds[idx])

    pred = (scores >= thr).astype(np.int8)

    return {
        "eer": eer,
        "eer_threshold": thr,
        "far_at_eer": far,
        "frr_at_eer": frr,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "f1_genuine_eer": float(
            f1_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "balanced_accuracy_eer": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "accuracy_eer": float(accuracy_score(y_true, pred)),
        "precision_genuine_eer": float(
            precision_score(y_true, pred, pos_label=1, zero_division=0)
        ),
        "recall_genuine_eer": float(
            recall_score(y_true, pred, pos_label=1, zero_division=0)
        ),
    }


def evaluate(
    model,
    family: str,
    candidate: str,
    strategy: str,
    fold: int,
    repeat: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    gpu_used: bool,
):
    y_train = train_df["user"].astype(str).eq(PROTECTED_USER).astype(np.int8).to_numpy()
    y_test = test_df["user"].astype(str).eq(PROTECTED_USER).astype(np.int8).to_numpy()

    t0 = time.perf_counter()
    model.fit(train_df[features], y_train)
    fit_s = time.perf_counter() - t0

    scores = score_genuine(model, test_df[features])
    m = metrics(y_test, scores)

    row = {
        "strategy": strategy,
        "candidate_family": family,
        "candidate": candidate,
        "fold": fold,
        "repeat": repeat,
        "gpu_used": gpu_used,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fit_seconds": fit_s,
        **m,
    }
    append_csv(pd.DataFrame([row]), RESULTS_PATH)

    pred = pd.DataFrame({
        "strategy": strategy,
        "candidate_family": family,
        "candidate": candidate,
        "fold": fold,
        "repeat": repeat,
        "session_id": test_df["session_id"].astype(str).to_numpy(),
        "window_id": test_df["window_id"].astype(str).to_numpy(),
        "actual_user": test_df["user"].astype(str).to_numpy(),
        "y_true_genuine": y_test,
        "score_genuine": scores,
    })
    append_csv(pred, PRED_PATH)

    del y_train, y_test, scores, pred
    gc.collect()
    return row


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "eer","far_at_eer","frr_at_eer","roc_auc","f1_genuine_eer",
        "balanced_accuracy_eer","accuracy_eer","precision_genuine_eer",
        "recall_genuine_eer","fit_seconds",
    ]

    rows = []
    for (strategy, family), g in results.groupby(
        ["strategy", "candidate_family"], sort=True
    ):
        row = {
            "strategy": strategy,
            "candidate_family": family,
            "candidate_examples": "|".join(
                sorted(g["candidate"].astype(str).unique())
            ),
            "gpu_used": bool(g["gpu_used"].any()),
            "n_results": len(g),
        }

        for metric in metric_cols:
            v = pd.to_numeric(g[metric], errors="coerce")
            row[f"{metric}_mean"] = float(v.mean())
            row[f"{metric}_std"] = float(v.std(ddof=0))
            row[f"{metric}_median"] = float(v.median())

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["strategy", "eer_mean", "roc_auc_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def choose_winners(summary: pd.DataFrame) -> pd.DataFrame:
    out = []
    for strategy, g in summary.groupby("strategy", sort=True):
        out.append(
            g.sort_values(
                [
                    "eer_mean",
                    "roc_auc_mean",
                    "balanced_accuracy_eer_mean",
                    "f1_genuine_eer_mean",
                ],
                ascending=[True, False, False, False],
            ).iloc[0].copy()
        )
    return pd.DataFrame(out).reset_index(drop=True)


def main():
    print("=" * 110)
    print("CYBELL-J — EXPERIMENTO 11B — CONTROLE DE QUANTIDADE")
    print("=" * 110)
    print(
        f"\nPyCaret: {pycaret.__version__}"
        f"\nCPU: {CPU_JOBS} threads"
        f"\nPyCaret GPU: False"
        f"\nRepetições pareadas: {N_REPEATS}"
        f"\nXGBoost GPU: leve ({XGB_TREES} árvores)"
        "\nRAM: chunks + float32 + gravação incremental"
    )

    reset_outputs()

    print("\n1) Testando CUDA...")
    use_gpu = gpu_available()
    print("   CUDA:", "disponível" if use_gpu else "indisponível; fallback CPU")

    print("\n2) Carregando 50/75...")
    df = load_data()
    features = feature_columns(df)
    df = compact(df, features)

    mouse = df[df["device_context"].astype(str).eq("mouse")].copy()

    print(f"   Total 50/75: {len(df):,}".replace(",", "."))
    print(f"   Mouse: {len(mouse):,}".replace(",", "."))
    print(f"   Features: {len(features)}")
    print("\n   Bianca por contexto:")
    print(
        df[df["user"].astype(str).eq(PROTECTED_USER)]
        .groupby("device_context").size().to_string()
    )

    folds = build_outer_folds(mouse)

    print("\n3) Folds externos:")
    for fd in folds:
        bianca_test = sorted(
            mouse.loc[
                mouse["user"].astype(str).eq(PROTECTED_USER)
                & mouse["session_id"].astype(str).isin(fd["test_sessions"]),
                "session_id",
            ].astype(str).unique()
        )
        print(f"   Fold {fd['fold']}: Bianca teste={','.join(bianca_test)}")

    rows = []
    py_registry = {}

    print("\n" + "=" * 110)
    print("4) EXECUTANDO CONTROLE PAREADO")
    print("=" * 110)

    for fd in folds:
        fold = fd["fold"]
        print("\n" + "-" * 110)
        print(f"FOLD {fold}")
        print("-" * 110)

        pools = fold_pools(df, mouse, fd)
        test = pools["test"]

        for rep in range(1, N_REPEATS + 1):
            print(f"\n   Repetição {rep}/{N_REPEATS}")

            unique_train, sub_train, comp = paired_train_sets(pools, fold, rep)
            append_csv(pd.DataFrame([comp]), COMPOSITION_PATH)

            datasets = {
                "PERFIL_UNICO_CONTROLADO": unique_train,
                "SUBPERFIL_MOUSE": sub_train,
            }

            for strategy, train in datasets.items():
                print(f"      {strategy}")

                key = (strategy, fold)
                if rep == 1:
                    try:
                        print("         PyCaret selecionando LR/RF/ET...")
                        pipe, name = pycaret_select(train, features, strategy, fold)
                        py_registry[key] = {"pipeline": pipe, "name": name}
                        print(f"         selecionado: {name}")
                    except Exception as exc:
                        py_registry[key] = None
                        print(f"         AVISO PyCaret: {type(exc).__name__}: {exc}")

                item = py_registry.get(key)
                if item is not None:
                    model = clone(item["pipeline"])
                    r = evaluate(
                        model=model,
                        family="PYCARET_SELECTED",
                        candidate=item["name"],
                        strategy=strategy,
                        fold=fold,
                        repeat=rep,
                        train_df=train,
                        test_df=test,
                        features=features,
                        gpu_used=False,
                    )
                    rows.append(r)
                    print(f"         PyCaret: EER={r['eer']:.4f} | AUC={r['roc_auc']:.4f}")
                    del model
                    gc.collect()

                yx = train["user"].astype(str).eq(PROTECTED_USER).astype(np.int8).to_numpy()
                xgb = build_xgb(yx, use_gpu)
                xgb_name = "XGBoost_GPU_Light" if use_gpu else "XGBoost_CPU_Light"

                r = evaluate(
                    model=xgb,
                    family="XGBOOST_LIGHT",
                    candidate=xgb_name,
                    strategy=strategy,
                    fold=fold,
                    repeat=rep,
                    train_df=train,
                    test_df=test,
                    features=features,
                    gpu_used=use_gpu,
                )
                rows.append(r)
                print(f"         XGBoost: EER={r['eer']:.4f} | AUC={r['roc_auc']:.4f}")

                del xgb, yx
                gc.collect()

            del unique_train, sub_train, datasets
            gc.collect()
            print("      RAM temporária liberada.")

        del test, pools
        gc.collect()

    results = pd.DataFrame(rows)
    summary = summarize(results)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    winners = choose_winners(summary)
    winners.to_csv(WINNERS_PATH, index=False, encoding="utf-8-sig")

    comparison = winners[
        [
            "strategy","candidate_family","candidate_examples","gpu_used",
            "n_results","eer_mean","eer_std","eer_median","far_at_eer_mean",
            "frr_at_eer_mean","roc_auc_mean","f1_genuine_eer_mean",
            "balanced_accuracy_eer_mean",
        ]
    ].sort_values(
        ["eer_mean", "roc_auc_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)

    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))
    comparison.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

    paired_rows = []
    for family, g in results.groupby("candidate_family", sort=True):
        wide = g.pivot_table(
            index=["fold", "repeat"],
            columns="strategy",
            values="eer",
            aggfunc="first",
        )
        if (
            "PERFIL_UNICO_CONTROLADO" in wide.columns
            and "SUBPERFIL_MOUSE" in wide.columns
        ):
            delta = (
                wide["PERFIL_UNICO_CONTROLADO"]
                - wide["SUBPERFIL_MOUSE"]
            ).dropna()

            paired_rows.append({
                "candidate_family": family,
                "n_pairs": len(delta),
                "eer_delta_unique_minus_subprofile_mean": float(delta.mean()),
                "eer_delta_unique_minus_subprofile_median": float(delta.median()),
                "unique_wins_lower_eer": int((delta < 0).sum()),
                "subprofile_wins_lower_eer": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "interpretation": (
                    "delta < 0 favorece PERFIL_UNICO_CONTROLADO; "
                    "delta > 0 favorece SUBPERFIL_MOUSE"
                ),
            })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(PAIRED_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 110)
    print("5) VENCEDOR POR ESTRATEGIA — QUANTIDADE CONTROLADA")
    print("=" * 110)
    print(
        winners[
            [
                "strategy","candidate_family","candidate_examples",
                "eer_mean","eer_std","eer_median","roc_auc_mean",
                "f1_genuine_eer_mean","balanced_accuracy_eer_mean",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 110)
    print("6) COMPARACAO FINAL")
    print("=" * 110)
    print(comparison.to_string(index=False))

    if not paired.empty:
        print("\n" + "=" * 110)
        print("7) COMPARACAO PAREADA")
        print("=" * 110)
        print(paired.to_string(index=False))

    best = comparison.iloc[0]
    print("\nMELHOR ESTRATEGIA CONTROLADA:")
    print(f"   {best['strategy']}")
    print(f"   EER medio: {best['eer_mean']:.4f}")
    print(f"   EER mediano: {best['eer_median']:.4f}")
    print(f"   AUC media: {best['roc_auc_mean']:.4f}")

    composition = pd.read_csv(COMPOSITION_PATH, low_memory=False)

    manifest = {
        "projeto": "CYBELL-J",
        "fase": "11B - controle por quantidade perfil unico x subperfil mouse",
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "pycaret_version": pycaret.__version__,
        "protected_user": PROTECTED_USER,
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP,
        "test_context": "mouse",
        "n_outer_folds": N_OUTER_FOLDS,
        "n_repeats": N_REPEATS,
        "same_test": True,
        "same_impostors": True,
        "same_genuine_count": True,
        "pycaret_models": PYCARET_MODELS,
        "resource_profile": {
            "cpu_jobs": CPU_JOBS,
            "pycaret_use_gpu": False,
            "xgboost_gpu_available": use_gpu,
            "xgboost_trees": XGB_TREES,
            "xgboost_depth": XGB_DEPTH,
            "xgboost_max_bin": XGB_MAX_BIN,
            "float32": True,
            "chunk_size": CHUNK,
            "incremental_outputs": True,
        },
        "sample_composition_mean": {
            "genuine_each_strategy": float(
                composition["n_genuine_each_strategy"].mean()
            ),
            "impostor_each_strategy": float(
                composition["n_impostor_each_strategy"].mean()
            ),
            "mixed_mouse": float(
                composition["mixed_genuine_mouse"].mean()
            ),
            "mixed_trackpad": float(
                composition["mixed_genuine_trackpad"].mean()
            ),
        },
        "best": {
            "strategy": str(best["strategy"]),
            "candidate_family": str(best["candidate_family"]),
            "eer_mean": float(best["eer_mean"]),
            "eer_median": float(best["eer_median"]),
            "roc_auc_mean": float(best["roc_auc_mean"]),
        },
        "notes": [
            "As duas estrategias usam exatamente o mesmo numero de genuinos.",
            "As duas estrategias usam exatamente a mesma amostra de impostores em cada fold/repeticao.",
            "As duas estrategias usam exatamente o mesmo conjunto de teste.",
            "O perfil unico controlado sorteia genuinos do pool Bianca mouse+trackpad.",
            "O subperfil usa somente Bianca mouse.",
            "PyCaret seleciona LR/RF/ExtraTrees apenas na repeticao 1 de cada fold/estrategia.",
            "As 5 repeticoes medem estabilidade de amostragem; nao sao 5 testes independentes.",
            "A comparacao entre dispositivos continua limitada a Bianca pela disponibilidade de trackpad.",
        ],
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 110)
    print("EXPERIMENTO 11B CONCLUIDO")
    print("=" * 110)
    print(f"\nResultados:\n{RESULTS_PATH}")
    print(f"\nResumo:\n{SUMMARY_PATH}")
    print(f"\nComparacao final:\n{COMPARISON_PATH}")
    print(f"\nComparacao pareada:\n{PAIRED_PATH}")
    print(f"\nComposicao:\n{COMPOSITION_PATH}")
    print(f"\nManifesto:\n{MANIFEST_PATH}")


if __name__ == "__main__":
    main()