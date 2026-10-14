"""
09_dispositivo_com_scroll.py

CYBELL-J — Experimento 09
Comparacao XGBoost SEM SCROLL x COM SCROLL, com agregacao temporal
em 5, 10, 15 e 20 segundos.

IMPORTANTE:
- device_context foi produzido por uma heuristica do front-end que usa wheel/scroll.
- SEM_SCROLL e a comparacao mais limpa para investigar sinal cinetico/comportamental.
- COM_SCROLL_OPERACIONAL mede desempenho operacional e NAO e validacao independente
  de hardware, pois existe circularidade parcial entre rotulo e features de scroll.
- O experimento continua intrausuaria (Bianca).
"""

from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "mouse_dataset" / "dataset_device_high_confidence.csv"
OUT = BASE_DIR / "data" / "mouse_training_device_scroll"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_FOLDS = OUT / "resultados_janela_por_fold.csv"
WINDOW_SUMMARY = OUT / "resumo_janela.csv"
WINDOW_PREDS = OUT / "predicoes_janela.csv"
TEMP_FOLDS = OUT / "resultados_temporais_por_fold.csv"
TEMP_SUMMARY = OUT / "resumo_temporal.csv"
TEMP_DECISIONS = OUT / "decisoes_temporais.csv"
FINAL_COMPARE = OUT / "comparacao_final.csv"
MODEL_PATH = OUT / "modelo_xgboost_com_scroll.joblib"
FEATURES_PATH = OUT / "features_com_scroll.json"
MANIFEST_PATH = OUT / "manifesto_scroll.json"

TARGET_USER = "bianca"
TARGET = "device_context"
POSITIVE = "trackpad"
WINDOW_SIZE = 25
OVERLAP = 75
HORIZONS = [5, 10, 15, 20]
N_FOLDS = 3
RANDOM_STATE = 42

META = {
    "target_user", "user", "session_id", "window_id", "window_size",
    "requested_overlap_pct", "step_events", "effective_overlap_pct",
    "window_start_ms", "window_end_ms", "n_events",
    "device_mouse_pct", "device_trackpad_pct", "device_context",
    "device_confidence", "device_high_confidence", "device_label_source",
    "target_user_device",
}
COMMON_EXCLUDED_NAMES = {"event_rate_hz", "duration_s"}
COMMON_EXCLUDED_PATTERNS = ("event_dt_s",)


def feature_sets(df):
    no_scroll, wheel = [], []
    for c in df.columns:
        if c in META or c in COMMON_EXCLUDED_NAMES:
            continue
        lc = c.lower()
        if any(p in lc for p in COMMON_EXCLUDED_PATTERNS):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if "wheel" in lc:
            wheel.append(c)
        else:
            no_scroll.append(c)
    if not no_scroll:
        raise RuntimeError("Nenhuma feature SEM scroll encontrada.")
    if not wheel:
        raise RuntimeError("Nenhuma feature contendo 'wheel' encontrada no dataset.")
    return no_scroll, no_scroll + wheel, wheel


def build_xgb(y):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw = neg / pos if pos else 1.0
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )),
    ])


def group_holdouts(df, n_splits=3):
    sessions = sorted(df["session_id"].astype(str).unique().tolist())
    if len(sessions) < 4:
        raise RuntimeError(f"Poucas sessoes: {len(sessions)}")
    global_rate = float((df[TARGET] == POSITIVE).mean())
    candidates = []
    test_sizes = sorted(set([
        max(1, math.floor(len(sessions) / 3)),
        max(2, math.ceil(len(sessions) / 3)),
        max(2, len(sessions) // 2),
    ]))
    for size in test_sizes:
        if size >= len(sessions):
            continue
        for tup in itertools.combinations(sessions, size):
            test_s = set(tup)
            train_s = set(sessions) - test_s
            tr = df[df["session_id"].astype(str).isin(train_s)]
            te = df[df["session_id"].astype(str).isin(test_s)]
            needed = {"mouse", "trackpad"}
            if not needed.issubset(set(tr[TARGET].astype(str).unique())):
                continue
            if not needed.issubset(set(te[TARGET].astype(str).unique())):
                continue
            tr_rate = float((tr[TARGET] == POSITIVE).mean())
            te_rate = float((te[TARGET] == POSITIVE).mean())
            frac = len(te) / len(df)
            score = abs(tr_rate-global_rate) + abs(te_rate-global_rate) + 0.5*abs(frac-0.35)
            candidates.append({
                "score": score,
                "train_groups": sorted(train_s),
                "test_groups": sorted(test_s),
            })
    if not candidates:
        raise RuntimeError("Nao foi possivel criar folds por sessao com as duas classes.")
    candidates.sort(key=lambda x: (x["score"], x["test_groups"]))
    out, seen = [], set()
    for c in candidates:
        k = tuple(c["test_groups"])
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= n_splits:
            break
    return out


def eer(y, score):
    fpr, tpr, thr = roc_curve(y, score, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr-fnr)))
    return float((fpr[i]+fnr[i])/2.0), float(thr[i])


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= 0.5).astype(int)
    e, et = eer(y, score)
    return {
        "n": len(y),
        "n_mouse": int((y == 0).sum()),
        "n_trackpad": int((y == 1).sum()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "eer": e,
        "eer_threshold": et,
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


def train_set(df, name, features, folds):
    rows, preds = [], []
    for fold_no, split in enumerate(folds, 1):
        tr = df[df["session_id"].astype(str).isin(split["train_groups"])].copy()
        te = df[df["session_id"].astype(str).isin(split["test_groups"])].copy()
        ytr = tr[TARGET].eq(POSITIVE).astype(int).to_numpy()
        yte = te[TARGET].eq(POSITIVE).astype(int).to_numpy()
        model = build_xgb(ytr)
        model.fit(tr[features], ytr)
        score = model.predict_proba(te[features])[:, 1]
        m = metrics(yte, score)
        rows.append({
            "feature_set": name,
            "fold": fold_no,
            "train_sessions": "|".join(split["train_groups"]),
            "test_sessions": "|".join(split["test_groups"]),
            "n_features": len(features),
            **m,
        })
        preds.append(pd.DataFrame({
            "feature_set": name,
            "fold": fold_no,
            "session_id": te["session_id"].astype(str).to_numpy(),
            "window_id": te["window_id"].astype(str).to_numpy(),
            "window_start_ms": pd.to_numeric(te["window_start_ms"], errors="coerce").to_numpy(),
            "window_end_ms": pd.to_numeric(te["window_end_ms"], errors="coerce").to_numpy(),
            "y_true": yte,
            "score_trackpad": score,
        }))
        print(f"   {name} | Fold {fold_no}: F1={m['f1']:.4f} | EER={m['eer']:.4f} | AUC={m['roc_auc']:.4f} | BalAcc={m['balanced_accuracy']:.4f}")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize_folds(df, keys):
    metric_cols = ["f1", "eer", "roc_auc", "balanced_accuracy", "accuracy", "precision", "recall"]
    rows = []
    for group_key, g in df.groupby(keys, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key))
        row["n_folds"] = len(g)
        if "n_features" in g:
            row["n_features"] = int(g["n_features"].iloc[0])
        for c in metric_cols:
            v = pd.to_numeric(g[c], errors="coerce")
            row[c+"_mean"] = float(v.mean())
            row[c+"_std"] = float(v.std(ddof=0))
        if "mean_predictions_per_decision" in g:
            v = pd.to_numeric(g["mean_predictions_per_decision"], errors="coerce")
            row["mean_predictions_per_decision_mean"] = float(v.mean())
        if "stable_fraction_pct" in g:
            v = pd.to_numeric(g["stable_fraction_pct"], errors="coerce")
            row["stable_fraction_pct_mean"] = float(v.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_group(g, horizon_s):
    g = g.sort_values(["window_end_ms", "window_start_ms", "window_id"]).reset_index(drop=True)
    times = g["window_end_ms"].to_numpy(dtype=float)
    scores = g["score_trackpad"].to_numpy(dtype=float)
    labels = g["y_true"].to_numpy(dtype=int)
    horizon_ms = horizon_s * 1000.0
    score_cs = np.concatenate([[0.0], np.cumsum(scores)])
    label_cs = np.concatenate([[0], np.cumsum(labels)])
    first_time = float(times[0])
    left = 0
    rows = []
    for i in range(len(g)):
        now = float(times[i])
        while left < i and now - times[left] > horizon_ms:
            left += 1
        n = i-left+1
        if n < 2:
            continue
        if now-first_time < horizon_ms:
            continue
        mean_score = float((score_cs[i+1]-score_cs[left])/n)
        sum_lab = int(label_cs[i+1]-label_cs[left])
        rows.append({
            "feature_set": g.loc[i, "feature_set"],
            "fold": int(g.loc[i, "fold"]),
            "session_id": str(g.loc[i, "session_id"]),
            "window_id": str(g.loc[i, "window_id"]),
            "horizon_s": horizon_s,
            "n_predictions_aggregated": n,
            "y_true": int(labels[i]),
            "score_single": float(scores[i]),
            "score_aggregated": mean_score,
            "stable_true_context": bool(sum_lab == 0 or sum_lab == n),
        })
    return pd.DataFrame(rows)


def temporal_decisions(preds):
    parts = []
    for h in HORIZONS:
        print(f"   Horizonte {h}s...")
        for _, g in preds.groupby(["feature_set", "fold", "session_id"], sort=True):
            a = aggregate_group(g, h)
            if not a.empty:
                parts.append(a)
    if not parts:
        raise RuntimeError("Nenhuma decisao temporal gerada.")
    return pd.concat(parts, ignore_index=True)


def temporal_eval(decisions):
    rows = []
    for (fs, h, fold), g in decisions.groupby(["feature_set", "horizon_s", "fold"], sort=True):
        m = metrics(g["y_true"], g["score_aggregated"])
        rows.append({
            "feature_set": fs,
            "horizon_s": h,
            "subset": "all",
            "fold": fold,
            "mean_predictions_per_decision": float(g["n_predictions_aggregated"].mean()),
            "stable_fraction_pct": float(g["stable_true_context"].mean()*100),
            **m,
        })
        stable = g[g["stable_true_context"]]
        if not stable.empty and stable["y_true"].nunique() == 2:
            ms = metrics(stable["y_true"], stable["score_aggregated"])
            rows.append({
                "feature_set": fs,
                "horizon_s": h,
                "subset": "stable_only",
                "fold": fold,
                "mean_predictions_per_decision": float(stable["n_predictions_aggregated"].mean()),
                "stable_fraction_pct": float(g["stable_true_context"].mean()*100),
                **ms,
            })
    return pd.DataFrame(rows)


def main():
    print("="*94)
    print("CYBELL-J — EXPERIMENTO 09 — SEM SCROLL x COM SCROLL")
    print("="*94)
    print("\nATENCAO: COM_SCROLL_OPERACIONAL tem circularidade parcial com o rotulo do front-end.")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    required = {"user", "session_id", "window_id", "window_size", "requested_overlap_pct", "window_start_ms", "window_end_ms", TARGET}
    missing = sorted(required-set(df.columns))
    if missing:
        raise RuntimeError(f"Colunas ausentes: {missing}")

    df = df[
        df["user"].astype(str).str.lower().eq(TARGET_USER)
        & df["window_size"].eq(WINDOW_SIZE)
        & df["requested_overlap_pct"].eq(OVERLAP)
        & df[TARGET].isin(["mouse", "trackpad"])
    ].copy()

    print(f"\n1) Janelas: {len(df):,} | Mouse: {(df[TARGET]=='mouse').sum():,} | Trackpad: {(df[TARGET]=='trackpad').sum():,} | Sessoes: {df['session_id'].nunique()}".replace(",", "."))

    no_scroll, with_scroll, wheel = feature_sets(df)
    print(f"\n2) SEM scroll: {len(no_scroll)} features")
    print(f"   Features wheel adicionadas: {len(wheel)}")
    print(f"   COM scroll: {len(with_scroll)} features")
    print("\n   Features wheel:")
    for c in wheel:
        print(f"   - {c}")

    folds = group_holdouts(df, N_FOLDS)
    print("\n3) Folds por sessao:")
    for i, f in enumerate(folds, 1):
        print(f"   Fold {i}: teste={','.join(f['test_groups'])}")

    print("\n4) Treinando XGBoost SEM scroll...")
    f0, p0 = train_set(df, "SEM_SCROLL", no_scroll, folds)
    print("\n5) Treinando XGBoost COM scroll...")
    f1, p1 = train_set(df, "COM_SCROLL_OPERACIONAL", with_scroll, folds)

    wf = pd.concat([f0, f1], ignore_index=True)
    wp = pd.concat([p0, p1], ignore_index=True)
    wf.to_csv(WINDOW_FOLDS, index=False, encoding="utf-8-sig")
    wp.to_csv(WINDOW_PREDS, index=False, encoding="utf-8-sig")

    ws = summarize_folds(wf, ["feature_set"])
    ws = ws.sort_values(["eer_mean", "roc_auc_mean", "balanced_accuracy_mean"], ascending=[True, False, False]).reset_index(drop=True)
    ws.to_csv(WINDOW_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n"+"="*94)
    print("RESULTADO — JANELA UNICA")
    print("="*94)
    print(ws[["feature_set", "n_features", "f1_mean", "eer_mean", "roc_auc_mean", "balanced_accuracy_mean"]].to_string(index=False))

    print("\n6) Agregacao temporal 5/10/15/20 s...")
    td = temporal_decisions(wp)
    td.to_csv(TEMP_DECISIONS, index=False, encoding="utf-8-sig")
    tf = temporal_eval(td)
    tf.to_csv(TEMP_FOLDS, index=False, encoding="utf-8-sig")
    ts = summarize_folds(tf, ["feature_set", "horizon_s", "subset"])
    ts = ts.sort_values(["subset", "eer_mean", "roc_auc_mean", "balanced_accuracy_mean"], ascending=[True, True, False, False]).reset_index(drop=True)
    ts.to_csv(TEMP_SUMMARY, index=False, encoding="utf-8-sig")

    tall = ts[ts["subset"].eq("all")].copy()
    print("\n"+"="*94)
    print("RESULTADO — AGREGAÇÃO TEMPORAL (TODAS AS DECISOES)")
    print("="*94)
    print(tall[["feature_set", "horizon_s", "f1_mean", "eer_mean", "roc_auc_mean", "balanced_accuracy_mean", "mean_predictions_per_decision_mean"]].to_string(index=False))

    rows = []
    for _, r in ws.iterrows():
        rows.append({
            "feature_set": r["feature_set"], "strategy": "single_window", "horizon_s": 0,
            "f1": r["f1_mean"], "eer": r["eer_mean"], "roc_auc": r["roc_auc_mean"],
            "balanced_accuracy": r["balanced_accuracy_mean"],
        })
    for _, r in tall.iterrows():
        rows.append({
            "feature_set": r["feature_set"], "strategy": "temporal_mean", "horizon_s": int(r["horizon_s"]),
            "f1": r["f1_mean"], "eer": r["eer_mean"], "roc_auc": r["roc_auc_mean"],
            "balanced_accuracy": r["balanced_accuracy_mean"],
        })
    comp = pd.DataFrame(rows).sort_values(["eer", "roc_auc", "balanced_accuracy", "f1"], ascending=[True, False, False, False]).reset_index(drop=True)
    comp.insert(0, "rank", np.arange(1, len(comp)+1))
    comp.to_csv(FINAL_COMPARE, index=False, encoding="utf-8-sig")

    best = comp.iloc[0]
    print("\n"+"="*94)
    print("MELHOR RESULTADO OPERACIONAL")
    print("="*94)
    print(f"\nFeature set: {best['feature_set']}")
    print(f"Estrategia: {best['strategy']}")
    print(f"Horizonte: {int(best['horizon_s'])} s")
    print(f"F1: {best['f1']:.4f}")
    print(f"EER: {best['eer']:.4f}")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")
    print(f"Balanced Accuracy: {best['balanced_accuracy']:.4f}")

    print("\n7) Salvando modelo XGBoost COM scroll em todas as janelas...")
    yfull = df[TARGET].eq(POSITIVE).astype(int).to_numpy()
    final_model = build_xgb(yfull)
    final_model.fit(df[with_scroll], yfull)
    joblib.dump(final_model, MODEL_PATH)

    with FEATURES_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "feature_set": "COM_SCROLL_OPERACIONAL",
            "base_features_without_scroll": no_scroll,
            "wheel_features_added": wheel,
            "features_final": with_scroll,
            "target": TARGET,
            "positive_class": POSITIVE,
            "window_size": WINDOW_SIZE,
            "overlap_pct": OVERLAP,
        }, f, ensure_ascii=False, indent=2)

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump({
            "projeto": "CYBELL-J",
            "fase": "Experimento 09 - SEM SCROLL x COM SCROLL",
            "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": DATASET_PATH.name,
            "target_user": TARGET_USER,
            "window_size": WINDOW_SIZE,
            "overlap_pct": OVERLAP,
            "model": "XGBoost",
            "horizons_seconds": HORIZONS,
            "features_no_scroll_count": len(no_scroll),
            "wheel_features_added_count": len(wheel),
            "features_with_scroll_count": len(with_scroll),
            "wheel_features_added": wheel,
            "best_result": {
                "feature_set": str(best["feature_set"]),
                "strategy": str(best["strategy"]),
                "horizon_s": int(best["horizon_s"]),
                "f1": float(best["f1"]),
                "eer": float(best["eer"]),
                "roc_auc": float(best["roc_auc"]),
                "balanced_accuracy": float(best["balanced_accuracy"]),
            },
            "methodological_notes": [
                "Mesmos folds por session_id para SEM_SCROLL e COM_SCROLL_OPERACIONAL.",
                "A unica diferenca entre os feature sets e a inclusao das colunas numericas contendo 'wheel'.",
                "Features de scroll participam da heuristica original que gerou device_context; COM_SCROLL_OPERACIONAL tem circularidade parcial.",
                "COM_SCROLL_OPERACIONAL nao e validacao independente de hardware.",
                "Agregacao temporal usa apenas previsoes passadas e a atual dentro da mesma sessao/fold.",
                "Experimento intrausuaria; nao demonstra generalizacao universal.",
            ],
        }, f, ensure_ascii=False, indent=2)

    print("\n"+"="*94)
    print("EXPERIMENTO 09 CONCLUIDO")
    print("="*94)
    print(f"\nResumo janela:\n{WINDOW_SUMMARY}")
    print(f"\nResumo temporal:\n{TEMP_SUMMARY}")
    print(f"\nComparacao final:\n{FINAL_COMPARE}")
    print(f"\nModelo com scroll:\n{MODEL_PATH}")
    print(f"\nManifesto:\n{MANIFEST_PATH}")


if __name__ == "__main__":
    main()