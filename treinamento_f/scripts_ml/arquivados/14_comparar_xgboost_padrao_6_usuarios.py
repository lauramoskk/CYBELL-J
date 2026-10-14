"""
14_comparar_xgboost_padrao_6_usuarios.py

CYBELL-J — ETAPA DE DECISÃO FINAL
COMPARAÇÃO CONTROLADA: UM ÚNICO ALGORITMO (XGBOOST) PARA OS 6 PERFIS

Objetivo
--------
Responder uma única pergunta:
"Vale a pena padronizar o sistema inteiro em XGBoost?"

Não abre busca de algoritmos.
Não usa PyCaret.
Não usa LSTM.
Não mistura trackpad.
Não altera os modelos finais já salvos.

Configuração fixa:
- dataset high-confidence
- device_context = mouse
- janela = 50 eventos
- overlap solicitado = 75% (usa o efetivo mais próximo, ex. 74%)
- EXATAMENTE 89 features do pipeline 10B
- 3 folds por session_id
- impostores de treino limitados a 2:1
- XGBoost leve fixo
- horizontes avaliados: 0, 5, 10, 15, 20 s
- seleção por menor EER médio com 3 folds válidos

Saídas:
data/xgboost_padrao_6_usuarios/
    resultados_folds_single.csv
    predicoes_oof.csv
    resultados_temporais_folds.csv
    resumo_temporal.csv
    melhor_horizonte_por_perfil.csv
    comparacao_com_modelos_personalizados.csv
    resultado_global.csv
    decisao_padronizacao.txt
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


CPU_THREADS = 2
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "mouse_dataset" / "dataset_device_high_confidence.csv"
CURRENT_FINAL_PATH = BASE_DIR / "data" / "modelos_finais_cybellj" / "configuracao_final.csv"
FINAL_FEATURES_PATH = BASE_DIR / "data" / "modelos_finais_cybellj" / "feature_names.json"

OUT = BASE_DIR / "data" / "xgboost_padrao_6_usuarios"
OUT.mkdir(parents=True, exist_ok=True)

FOLDS_PATH = OUT / "resultados_folds_single.csv"
PRED_PATH = OUT / "predicoes_oof.csv"
TEMP_FOLDS_PATH = OUT / "resultados_temporais_folds.csv"
TEMP_SUMMARY_PATH = OUT / "resumo_temporal.csv"
BEST_PATH = OUT / "melhor_horizonte_por_perfil.csv"
COMPARE_PATH = OUT / "comparacao_com_modelos_personalizados.csv"
GLOBAL_PATH = OUT / "resultado_global.csv"
DECISION_PATH = OUT / "decisao_padronizacao.txt"

RANDOM_STATE = 42
N_FOLDS = 3
WINDOW_SIZE = 50
OVERLAP_TARGET = 75.0
HORIZONS = [0, 5, 10, 15, 20]
IMPOSTOR_RATIO = 2.0
MIN_WINDOWS_TEMP = 2
REQUIRE_FULL_HORIZON = True

USERS = [
    "Julia104soup",
    "bianca",
    "lala",
    "laura_oliveira",
    "lina",
    "yass",
]

METADATA_COLUMNS = {
    "target_user",
    "user",
    "session_id",
    "window_id",
    "window_size",
    "requested_overlap_pct",
    "step_events",
    "effective_overlap_pct",
    "device_context",
    "device_confidence",
    "high_confidence",
    "target_user_device",
    "mouse_pct",
    "trackpad_pct",
    "window_start_ms",
    "window_end_ms",
    "window_start_timestamp",
    "window_end_timestamp",
    "start_timestamp",
    "end_timestamp",
    "start_time",
    "end_time",
}

def stable_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (RANDOM_STATE + int(h[:8], 16)) % (2**32 - 1)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def find_col(columns, candidates):
    m = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    return None

def normalize_overlap(series):
    v = pd.to_numeric(series, errors="coerce")
    finite = v[np.isfinite(v)]
    if finite.empty:
        return v
    if float(np.nanmedian(np.abs(finite))) <= 1.5:
        return v * 100.0
    return v

def detect_time_col(df):
    candidates = [
        "window_end_ms", "window_end_timestamp", "end_timestamp",
        "window_end_time", "end_time", "timestamp_end", "ts_end", "end_ts",
        "window_start_ms", "window_start_timestamp", "start_timestamp",
        "window_start_time", "start_time", "timestamp_start", "ts_start", "start_ts",
    ]
    col = find_col(df.columns, candidates)
    if col:
        return col
    for c in df.columns:
        lc = str(c).lower()
        if ("end" in lc or "start" in lc) and ("time" in lc or "timestamp" in lc or lc.endswith("_ts")):
            return c
    return None

def time_to_ms(series):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise RuntimeError("Coluna temporal sem valores válidos.")
    med = float(np.nanmedian(np.abs(finite)))
    if med > 1e17:
        factor = 1e-6
    elif med > 1e14:
        factor = 1e-3
    elif med > 1e11:
        factor = 1.0
    elif med > 1e8:
        factor = 1000.0
    else:
        factor = 1000.0
    return np.rint(x * factor)

def load_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH, low_memory=False)

    user_col = find_col(df.columns, ["user", "username"])
    session_col = find_col(df.columns, ["session_id", "session"])
    window_col = find_col(df.columns, ["window_size", "window_events", "n_events"])
    overlap_col = find_col(df.columns, ["effective_overlap_pct", "overlap_percent", "overlap_pct", "overlap"])
    device_col = find_col(df.columns, ["device_context", "window_device", "context_device"])

    if user_col is None or session_col is None:
        raise RuntimeError("user/session_id não encontrados.")

    if window_col is not None:
        w = pd.to_numeric(df[window_col], errors="coerce")
        df = df.loc[w.eq(WINDOW_SIZE)].copy()

    if overlap_col is not None:
        ov = normalize_overlap(df[overlap_col])
        available = np.sort(pd.Series(ov).dropna().unique().astype(float))
        if len(available) == 0:
            raise RuntimeError("Sem overlap válido.")
        nearest = float(available[np.argmin(np.abs(available - OVERLAP_TARGET))])
        if abs(nearest - OVERLAP_TARGET) > 5:
            raise RuntimeError(f"Overlap efetivo incompatível: {nearest}")
        df = df.loc[np.isclose(ov, nearest, atol=1e-9)].copy()
        print(f"Overlap efetivo usado: {nearest:g}%")

    if device_col is not None:
        dev = df[device_col].astype(str).str.strip().str.lower()
        df = df.loc[dev.eq("mouse")].copy()

    df[user_col] = df[user_col].astype(str).str.strip()
    df[session_col] = df[session_col].astype(str).str.strip()
    df = df.loc[df[user_col].isin(USERS)].copy()

    time_col = detect_time_col(df)
    if time_col is None:
        raise RuntimeError("Coluna temporal não encontrada.")

    df["_time_ms"] = time_to_ms(df[time_col])
    df = df.dropna(subset=["_time_ms"]).copy()
    df["_time_ms"] = df["_time_ms"].astype("int64")
    df = df.sort_values([user_col, session_col, "_time_ms"]).reset_index(drop=True)

    return df, user_col, session_col, time_col

def select_features(df, user_col, session_col, time_col):
    """
    Usa a lista EXATA de 89 features já validada e salva pela Fase 13.
    Isso evita qualquer divergência futura na lógica de exclusão de metadata.
    """
    if not FINAL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Lista final de features não encontrada:\n{FINAL_FEATURES_PATH}\n"
            "Execute primeiro a Fase 13 corrigida (89 features)."
        )

    with FINAL_FEATURES_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    feats = payload.get("features", [])
    n_features = payload.get("n_features", len(feats))

    if n_features != 89 or len(feats) != 89:
        raise RuntimeError(
            f"O feature_names.json deveria conter exatamente 89 features, "
            f"mas informa n_features={n_features} e lista={len(feats)}."
        )

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError(
            "As seguintes features finais não existem no dataset atual:\n"
            + "\n".join(missing)
        )

    forbidden = {
        "requested_overlap_pct",
        "step_events",
        "effective_overlap_pct",
        "window_size",
        "window_id",
        "device_context",
        "device_confidence",
        "high_confidence",
    }

    leaked = sorted(set(feats) & forbidden)
    if leaked:
        raise RuntimeError(
            "A lista final de 89 features contém metadata indevida: "
            + ", ".join(leaked)
        )

    print(
        "Features: 89 — carregadas diretamente de "
        "data/modelos_finais_cybellj/feature_names.json."
    )
    print(
        "Consistência garantida com a Fase 13; nenhuma redescoberta "
        "automática de features será feita."
    )
    return feats

def numeric_matrix(df, features):
    X = (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    med = X.median(axis=0, numeric_only=True)
    X = X.fillna(med).fillna(0.0).astype(np.float32)
    return X

def build_folds(df, user_col, session_col):
    fold_sessions = [set() for _ in range(N_FOLDS)]

    for user in USERS:
        sessions = sorted(
            df.loc[df[user_col].eq(user), session_col].astype(str).unique().tolist()
        )
        if len(sessions) < N_FOLDS:
            raise RuntimeError(f"{user}: apenas {len(sessions)} sessões.")
        rng = np.random.default_rng(stable_seed(f"folds|{user}"))
        rng.shuffle(sessions)
        for i, s in enumerate(sessions):
            fold_sessions[i % N_FOLDS].add(s)

    all_sessions = set(df[session_col].astype(str).unique())
    return [
        {
            "fold": i + 1,
            "test_sessions": fold_sessions[i],
            "train_sessions": all_sessions - fold_sessions[i],
        }
        for i in range(N_FOLDS)
    ]

def sample_impostors(df, idx, n_genuine, protected_user, tag, user_col):
    target = int(round(n_genuine * IMPOSTOR_RATIO))
    if len(idx) <= target:
        return idx

    pool = df.loc[idx, [user_col]].copy()
    pool["_idx"] = idx

    parts = []
    remaining = target
    imp_users = sorted(pool[user_col].astype(str).unique())

    for pos, u in enumerate(imp_users):
        uidx = pool.loc[pool[user_col].astype(str).eq(u), "_idx"].to_numpy(dtype=np.int64)
        left_groups = len(imp_users) - pos
        take = min(len(uidx), max(1, remaining // left_groups))
        rng = np.random.default_rng(stable_seed(f"imp|{protected_user}|{tag}|{u}"))
        parts.append(rng.choice(uidx, size=take, replace=False))
        remaining -= take

    out = np.concatenate(parts).astype(np.int64)
    if len(out) > target:
        rng = np.random.default_rng(stable_seed(f"trim|{protected_user}|{tag}"))
        out = rng.choice(out, size=target, replace=False)
    return out

def build_xgb(protected_user, fold):
    return XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.90,
        colsample_bytree=0.90,
        max_bin=128,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=stable_seed(f"xgb|{protected_user}|{fold}"),
        n_jobs=CPU_THREADS,
        tree_method="hist",
        device="cpu",
        verbosity=0,
    )

def eer(y, s):
    fpr, tpr, th = roc_curve(y, s, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(th[i])

def metrics(y, s):
    e, th = eer(y, s)
    p = (np.asarray(s) >= th).astype(np.int8)
    return {
        "eer": e,
        "threshold": th,
        "auc": float(roc_auc_score(y, s)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
    }

def aggregate_session(g, horizon_s):
    g = g.sort_values(["_time_ms", "_row_index"]).reset_index(drop=True)
    t = g["_time_ms"].to_numpy(dtype=np.int64)
    s = g["score_genuine"].to_numpy(dtype=np.float64)
    y = g["y_true_genuine"].to_numpy(dtype=np.int8)

    if horizon_s == 0:
        return pd.DataFrame({"y": y, "score": s, "n": np.ones(len(g), dtype=np.int16)})

    h = horizon_s * 1000
    cs = np.concatenate([[0.0], np.cumsum(s, dtype=np.float64)])
    first = int(t[0])
    left = 0
    oy, oscore, on = [], [], []

    for i in range(len(g)):
        now = int(t[i])
        while left < i and now - int(t[left]) > h:
            left += 1
        n = i - left + 1
        if n < MIN_WINDOWS_TEMP:
            continue
        if REQUIRE_FULL_HORIZON and now - first < h:
            continue
        oy.append(int(y[i]))
        oscore.append(float((cs[i + 1] - cs[left]) / n))
        on.append(n)

    return pd.DataFrame({"y": oy, "score": oscore, "n": on})

def temporal_evaluate(pred):
    rows = []

    for (user, fold), fdf in pred.groupby(["protected_user", "fold"], sort=True):
        for h in HORIZONS:
            parts = []
            for _, s in fdf.groupby("session_id", sort=False):
                part = aggregate_session(s, h)
                if not part.empty:
                    parts.append(part)

            if not parts:
                continue

            dec = pd.concat(parts, ignore_index=True)
            if dec["y"].nunique() < 2:
                continue

            m = metrics(dec["y"].to_numpy(), dec["score"].to_numpy())
            rows.append({
                "protected_user": user,
                "fold": int(fold),
                "horizon_s": int(h),
                "n_decisions": len(dec),
                **m,
            })

    return pd.DataFrame(rows)

def summarize_temporal(tf):
    rows = []
    for (u, h), g in tf.groupby(["protected_user", "horizon_s"], sort=True):
        rows.append({
            "protected_user": u,
            "horizon_s": int(h),
            "n_valid_folds": int(g["fold"].nunique()),
            "eer_mean": float(g["eer"].mean()),
            "eer_std": float(g["eer"].std(ddof=0)),
            "auc_mean": float(g["auc"].mean()),
            "f1_mean": float(g["f1"].mean()),
            "balanced_accuracy_mean": float(g["balanced_accuracy"].mean()),
        })
    return pd.DataFrame(rows)

def choose_best(summary):
    out = []
    for u, g in summary.groupby("protected_user", sort=True):
        eligible = g.loc[g["n_valid_folds"].eq(N_FOLDS)].copy()
        if eligible.empty:
            raise RuntimeError(f"{u}: nenhum horizonte com 3 folds válidos.")
        best = eligible.sort_values(
            ["eer_mean", "auc_mean", "f1_mean", "horizon_s"],
            ascending=[True, False, False, True],
        ).iloc[0]
        out.append(best)
    return pd.DataFrame(out).reset_index(drop=True)

def main():
    print("=" * 104)
    print("CYBELL-J — COMPARAÇÃO FINAL: XGBOOST PADRÃO NOS 6 PERFIS")
    print("=" * 104)
    print("\nMouse-only. 89 features EXATAS da Fase 13. Sem PyCaret. Sem LSTM. Sem alterar modelos finais.\n")

    seed_all(RANDOM_STATE)
    df, user_col, session_col, time_col = load_data()
    features = select_features(df, user_col, session_col, time_col)
    X = numeric_matrix(df, features)
    folds = build_folds(df, user_col, session_col)

    print("\nBase usada:")
    print(
        df.groupby(user_col)
        .agg(windows=(session_col, "size"), sessions=(session_col, "nunique"))
        .reset_index()
        .to_string(index=False)
    )

    fold_rows = []
    pred_parts = []

    for pos, u in enumerate(USERS, start=1):
        print(f"\n[{pos}/6] {u}")

        for fd in folds:
            fold = fd["fold"]
            train_mask = df[session_col].astype(str).isin(fd["train_sessions"])
            test_mask = df[session_col].astype(str).isin(fd["test_sessions"])

            genuine_idx = np.flatnonzero((train_mask & df[user_col].eq(u)).to_numpy()).astype(np.int64)
            imp_pool_idx = np.flatnonzero((train_mask & ~df[user_col].eq(u)).to_numpy()).astype(np.int64)
            test_idx = np.flatnonzero(test_mask.to_numpy()).astype(np.int64)

            imp_idx = sample_impostors(df, imp_pool_idx, len(genuine_idx), u, f"fold{fold}", user_col)
            train_idx = np.concatenate([genuine_idx, imp_idx])

            rng = np.random.default_rng(stable_seed(f"shuffle|{u}|fold{fold}"))
            rng.shuffle(train_idx)

            y_train = df.loc[train_idx, user_col].eq(u).astype(np.int8).to_numpy()
            y_test = df.loc[test_idx, user_col].eq(u).astype(np.int8).to_numpy()

            model = build_xgb(u, fold)
            model.fit(X.iloc[train_idx], y_train)
            scores = model.predict_proba(X.iloc[test_idx])[:, 1].astype(np.float64)

            m = metrics(y_test, scores)
            fold_rows.append({
                "protected_user": u,
                "fold": fold,
                "eer_single": m["eer"],
                "auc_single": m["auc"],
                "f1_single": m["f1"],
                "balanced_accuracy_single": m["balanced_accuracy"],
            })

            pred_parts.append(pd.DataFrame({
                "protected_user": u,
                "fold": fold,
                "session_id": df.loc[test_idx, session_col].astype(str).to_numpy(),
                "actual_user": df.loc[test_idx, user_col].astype(str).to_numpy(),
                "_time_ms": df.loc[test_idx, "_time_ms"].to_numpy(dtype=np.int64),
                "_row_index": test_idx,
                "y_true_genuine": y_test,
                "score_genuine": scores,
            }))

            print(
                f"   fold {fold}: EER={m['eer']:.4f} "
                f"AUC={m['auc']:.4f} F1={m['f1']:.4f}"
            )

            del model, scores, y_train, y_test, train_idx, imp_idx, genuine_idx, imp_pool_idx, test_idx
            gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(FOLDS_PATH, index=False, encoding="utf-8-sig")

    pred = pd.concat(pred_parts, ignore_index=True)
    pred.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

    tf = temporal_evaluate(pred)
    tf.to_csv(TEMP_FOLDS_PATH, index=False, encoding="utf-8-sig")

    summary = summarize_temporal(tf)
    summary.to_csv(TEMP_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    best = choose_best(summary)
    best.to_csv(BEST_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 104)
    print("MELHOR XGBOOST PADRÃO POR PERFIL")
    print("=" * 104)
    print(
        best[
            [
                "protected_user",
                "horizon_s",
                "eer_mean",
                "auc_mean",
                "f1_mean",
                "balanced_accuracy_mean",
            ]
        ].to_string(index=False)
    )

    comparison = pd.DataFrame()

    if CURRENT_FINAL_PATH.exists():
        current = pd.read_csv(CURRENT_FINAL_PATH, low_memory=False)
        wanted = [
            "protected_user",
            "candidate",
            "horizon_s",
            "eer_oof_calibration",
            "auc_oof_calibration",
            "f1_oof_calibration",
        ]
        current = current[[c for c in wanted if c in current.columns]].copy()
        current = current.rename(
            columns={
                "candidate": "personalized_candidate",
                "horizon_s": "personalized_horizon_s",
                "eer_oof_calibration": "personalized_eer",
                "auc_oof_calibration": "personalized_auc",
                "f1_oof_calibration": "personalized_f1",
            }
        )

        xgb = best.rename(
            columns={
                "horizon_s": "xgb_horizon_s",
                "eer_mean": "xgb_eer",
                "auc_mean": "xgb_auc",
                "f1_mean": "xgb_f1",
                "balanced_accuracy_mean": "xgb_balanced_accuracy",
            }
        )

        comparison = xgb.merge(current, on="protected_user", how="left")
        comparison["eer_delta_xgb_minus_personalized"] = (
            comparison["xgb_eer"] - comparison["personalized_eer"]
        )
        comparison["winner_eer"] = np.where(
            comparison["xgb_eer"] < comparison["personalized_eer"],
            "XGBOOST_PADRAO",
            np.where(
                comparison["xgb_eer"] > comparison["personalized_eer"],
                "PERSONALIZADO",
                "EMPATE",
            ),
        )
        comparison.to_csv(COMPARE_PATH, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 104)
        print("XGBOOST PADRÃO x MODELOS PERSONALIZADOS ATUAIS")
        print("=" * 104)
        print(
            comparison[
                [
                    "protected_user",
                    "xgb_horizon_s",
                    "xgb_eer",
                    "personalized_candidate",
                    "personalized_horizon_s",
                    "personalized_eer",
                    "eer_delta_xgb_minus_personalized",
                    "winner_eer",
                ]
            ].to_string(index=False)
        )

    global_row = {
        "algorithm": "XGBoost_standardized",
        "profiles": int(best["protected_user"].nunique()),
        "eer_mean_profiles": float(best["eer_mean"].mean()),
        "eer_median_profiles": float(best["eer_mean"].median()),
        "auc_mean_profiles": float(best["auc_mean"].mean()),
        "f1_mean_profiles": float(best["f1_mean"].mean()),
        "balanced_accuracy_mean_profiles": float(best["balanced_accuracy_mean"].mean()),
    }

    if not comparison.empty and "personalized_eer" in comparison.columns:
        global_row["personalized_eer_mean_profiles"] = float(comparison["personalized_eer"].mean())
        global_row["eer_delta_xgb_minus_personalized_global"] = (
            global_row["eer_mean_profiles"]
            - global_row["personalized_eer_mean_profiles"]
        )

    pd.DataFrame([global_row]).to_csv(GLOBAL_PATH, index=False, encoding="utf-8-sig")

    if not comparison.empty:
        delta = float(global_row["eer_delta_xgb_minus_personalized_global"])
        wins = int((comparison["winner_eer"] == "XGBOOST_PADRAO").sum())
        losses = int((comparison["winner_eer"] == "PERSONALIZADO").sum())

        # Regra prática conservadora: padronizar se não piorar > 0.02 EER absoluto
        # e não houver perda forte (>0.05) em algum perfil.
        worst_loss = float(comparison["eer_delta_xgb_minus_personalized"].max())
        standardize = (delta <= 0.02) and (worst_loss <= 0.05)

        decision = (
            f"DECISÃO: {'PADRONIZAR EM XGBOOST' if standardize else 'MANTER MODELOS PERSONALIZADOS'}\n"
            f"EER médio XGBoost: {global_row['eer_mean_profiles']:.6f}\n"
            f"EER médio personalizado: {global_row['personalized_eer_mean_profiles']:.6f}\n"
            f"Delta global XGBoost - personalizado: {delta:+.6f}\n"
            f"Vitórias XGBoost: {wins}/6\n"
            f"Vitórias personalizado: {losses}/6\n"
            f"Pior perda individual do XGBoost: {worst_loss:+.6f}\n"
            "\nObservação: este teste é mouse-only. Trackpad continua fora desta decisão.\n"
        )
    else:
        decision = (
            "Não foi possível comparar automaticamente com configuracao_final.csv.\n"
            "O resultado XGBoost foi salvo para análise manual.\n"
        )

    DECISION_PATH.write_text(decision, encoding="utf-8")
    print("\n" + "=" * 104)
    print(decision)
    print("=" * 104)
    print(f"\nArquivos em:\n{OUT}")

if __name__ == "__main__":
    main()