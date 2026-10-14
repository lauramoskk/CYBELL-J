"""
12_treinar_lstm_tensorflow.py

CYBELL-J — FASE 12
TREINAMENTO LSTM PARA AUTENTICAÇÃO CONTÍNUA 1:1
VERSÃO TENSORFLOW / KERAS — WINDOWS NATIVO / CPU

Por que esta versão?
--------------------
TensorFlow moderno não usa GPU NVIDIA oficialmente no Windows nativo
(após TF 2.10). Para evitar WSL2/CUDA/PyTorch e reduzir problemas de
ambiente, esta versão usa TensorFlow/Keras em CPU.

O modelo é pequeno e conservador:
- LSTM(32)
- Dense(16)
- batch 32
- máximo 12 épocas
- early stopping pela loss de treino
- CPU limitada a 2 threads
- float32
- 3 folds por session_id
- autenticação 1:1 para 6 usuárias

Metodologia
-----------
- Snapshot bruto de mouse.
- Somente movimentos rotulados como mouse.
- Sessão nova após gap > 300 s.
- Sequências de 50 movimentos.
- Stride 12 (~76% overlap).
- Quebra de continuidade quando gap entre movimentos > 2 s.
- Features por movimento:
    dx
    dy
    dt_s
    distance
    log_speed
    signed_log_acceleration
    sin_direction
    cos_direction
- Normalização calculada SOMENTE no treino de cada fold.
- Impostores de treino limitados a 2:1 em relação aos genuínos.
- Teste usa todas as sequências das sessões externas.
- Nenhuma session_id aparece em treino e teste ao mesmo tempo.

PyCaret
-------
PyCaret não treina a LSTM. Ele permanece como baseline clássico.
Ao final, este script procura automaticamente:
data/mouse_authentication_temporal/melhor_horizonte_por_perfil.csv
e compara LSTM x clássico/PyCaret.

Saídas
------
data/lstm_tensorflow_authentication/
    sequence_cache/
    modelos/
    resultados_folds.csv
    predicoes_oof.csv
    resumo_single.csv
    resultados_temporais_folds.csv
    resumo_temporal.csv
    melhor_horizonte_por_perfil.csv
    comparacao_lstm_vs_classico.csv
    resultado_global.csv
    manifesto_12_tensorflow.json
"""

from __future__ import annotations

# ============================================================
# RECURSOS — CONFIGURAR ANTES DE IMPORTAR TENSORFLOW
# ============================================================

import os

CPU_THREADS = 2

os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(CPU_THREADS))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import gc
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "\nERRO: TensorFlow não está instalado neste ambiente.\n"
        "Crie/ative .venv_tf e execute:\n"
        "python -m pip install tensorflow==2.21.0 pandas scikit-learn\n"
    ) from exc

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = (
    BASE_DIR
    / "data"
    / "mouse_snapshot"
    / "mouse_events_raw.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "lstm_tensorflow_authentication"
)

CACHE_DIR = (
    OUTPUT_DIR
    / "sequence_cache"
)

MODELS_DIR = (
    OUTPUT_DIR
    / "modelos"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = CACHE_DIR / "X_sequences.npy"
META_PATH = CACHE_DIR / "metadata_sequences.csv"
CACHE_MANIFEST_PATH = CACHE_DIR / "cache_manifest.json"

FOLD_RESULTS_PATH = OUTPUT_DIR / "resultados_folds.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predicoes_oof.csv"
SINGLE_SUMMARY_PATH = OUTPUT_DIR / "resumo_single.csv"
TEMPORAL_FOLDS_PATH = OUTPUT_DIR / "resultados_temporais_folds.csv"
TEMPORAL_SUMMARY_PATH = OUTPUT_DIR / "resumo_temporal.csv"
BEST_TEMPORAL_PATH = OUTPUT_DIR / "melhor_horizonte_por_perfil.csv"
COMPARISON_PATH = OUTPUT_DIR / "comparacao_lstm_vs_classico.csv"
GLOBAL_PATH = OUTPUT_DIR / "resultado_global.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifesto_12_tensorflow.json"

CLASSIC_BASELINE_PATH = (
    BASE_DIR
    / "data"
    / "mouse_authentication_temporal"
    / "melhor_horizonte_por_perfil.csv"
)


# ============================================================
# CONFIGURAÇÃO CIENTÍFICA
# ============================================================

RANDOM_STATE = 42

SESSION_GAP_MS = 300_000
MOVE_CONTINUITY_GAP_MS = 2_000

SEQUENCE_LENGTH = 50
SEQUENCE_STRIDE = 12
N_FOLDS = 3

IMPOSTOR_TO_GENUINE_TRAIN_RATIO = 2.0

FEATURE_NAMES = [
    "dx",
    "dy",
    "dt_s",
    "distance",
    "log_speed",
    "signed_log_acceleration",
    "sin_direction",
    "cos_direction",
]

HIDDEN_SIZE = 32
DENSE_SIZE = 16
DROPOUT = 0.20

BATCH_SIZE = 32
MAX_EPOCHS = 12
MIN_DELTA = 1e-4
EARLY_STOPPING_PATIENCE = 3

LEARNING_RATE = 1e-3

TEMPORAL_HORIZONS_SECONDS = [0, 5, 10, 15, 20]
TEMPORAL_REQUIRE_FULL_HORIZON = True
TEMPORAL_MIN_SCORES = 2

MAX_DT_SECONDS = 2.0
MAX_ABS_DXY = 2000.0
MAX_DISTANCE = 3000.0
MAX_SPEED = 50_000.0
MAX_ABS_ACCEL = 1_000_000.0


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (RANDOM_STATE + int(digest[:8], 16)) % (2**32 - 1)


def seed_everything(value: int):
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)


def configure_tensorflow():
    try:
        tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    # Esta versão é deliberadamente CPU-only.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


# ============================================================
# OUTPUT
# ============================================================

def reset_outputs():
    for path in [
        FOLD_RESULTS_PATH,
        PREDICTIONS_PATH,
        TEMPORAL_FOLDS_PATH,
    ]:
        if path.exists():
            path.unlink()


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


# ============================================================
# CACHE DE SEQUÊNCIAS
# ============================================================

def movement_mask(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str).str.strip().str.lower()
    )
    return normalized.isin([
        "move",
        "mousemove",
        "mouse_move",
        "pointermove",
    ])


def load_raw_compact() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Snapshot bruto não encontrado:\n{RAW_PATH}"
        )

    header = pd.read_csv(RAW_PATH, nrows=0)

    required = {
        "user",
        "timestamp",
        "event_type",
        "device_type",
        "x",
        "y",
    }

    missing = sorted(required - set(header.columns))
    if missing:
        raise RuntimeError(
            f"Colunas ausentes no CSV bruto: {missing}\n"
            f"Encontradas: {header.columns.tolist()}"
        )

    usecols = [
        "user",
        "timestamp",
        "event_type",
        "device_type",
        "x",
        "y",
    ]

    parts = []

    for chunk in pd.read_csv(
        RAW_PATH,
        usecols=usecols,
        chunksize=75_000,
        low_memory=False,
    ):
        for col in ["timestamp", "x", "y"]:
            chunk[col] = pd.to_numeric(
                chunk[col],
                errors="coerce",
            )

        chunk = chunk.dropna(
            subset=["user", "timestamp"]
        )

        parts.append(chunk)

    if not parts:
        raise RuntimeError("Snapshot bruto vazio.")

    raw = pd.concat(parts, ignore_index=True)

    del parts
    gc.collect()

    raw["timestamp"] = raw["timestamp"].astype("int64")
    raw["user"] = raw["user"].astype("string")
    raw["event_type"] = raw["event_type"].astype("string")
    raw["device_type"] = raw["device_type"].astype("string")

    raw = raw.sort_values(
        ["user", "timestamp"]
    ).reset_index(drop=True)

    return raw


def assign_sessions(raw: pd.DataFrame) -> pd.DataFrame:
    session_number = np.zeros(
        len(raw),
        dtype=np.int16,
    )

    for _, indices in raw.groupby(
        "user",
        sort=False,
    ).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        times = raw.loc[idx, "timestamp"].to_numpy(dtype=np.int64)

        start = np.ones(len(idx), dtype=bool)
        if len(idx) > 1:
            start[1:] = np.diff(times) > SESSION_GAP_MS

        session_number[idx] = np.cumsum(
            start,
            dtype=np.int16,
        )

    raw["_session_number"] = session_number

    raw["session_id"] = (
        raw["user"].astype(str)
        + "_S"
        + raw["_session_number"]
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )

    return raw


def segment_features(segment: pd.DataFrame):
    if len(segment) < SEQUENCE_LENGTH + 1:
        return None

    times = segment["timestamp"].to_numpy(dtype=np.int64)
    x = segment["x"].to_numpy(dtype=np.float64)
    y = segment["y"].to_numpy(dtype=np.float64)

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(times) / 1000.0

    dt = np.where(dt > 0, dt, 0.001)

    dx = np.clip(dx, -MAX_ABS_DXY, MAX_ABS_DXY)
    dy = np.clip(dy, -MAX_ABS_DXY, MAX_ABS_DXY)
    dt = np.clip(dt, 0.001, MAX_DT_SECONDS)

    distance = np.sqrt(dx * dx + dy * dy)
    distance = np.clip(distance, 0.0, MAX_DISTANCE)

    speed = distance / dt
    speed = np.clip(speed, 0.0, MAX_SPEED)

    acceleration = np.zeros_like(speed)

    if len(speed) > 1:
        acceleration[1:] = np.diff(speed) / dt[1:]

    acceleration = np.clip(
        acceleration,
        -MAX_ABS_ACCEL,
        MAX_ABS_ACCEL,
    )

    direction = np.arctan2(dy, dx)

    features = np.column_stack([
        dx,
        dy,
        dt,
        distance,
        np.log1p(speed),
        np.sign(acceleration) * np.log1p(np.abs(acceleration)),
        np.sin(direction),
        np.cos(direction),
    ]).astype(np.float32)

    return features, times[1:]


def create_sequence_cache():
    print("\nCriando cache de sequências...")

    raw = load_raw_compact()

    print(
        f"   eventos brutos: {len(raw):,}".replace(",", ".")
    )

    raw = assign_sessions(raw)

    is_move = (
        movement_mask(raw["event_type"])
        & raw["device_type"].astype(str).str.lower().eq("mouse")
        & raw["x"].notna()
        & raw["y"].notna()
    )

    move = raw.loc[
        is_move,
        [
            "user",
            "session_id",
            "timestamp",
            "x",
            "y",
        ],
    ].copy()

    print(
        f"   movimentos mouse: {len(move):,}".replace(",", ".")
    )

    del raw
    gc.collect()

    sequences = []
    meta_rows = []
    sequence_index = 0

    for (user, session_id), session in move.groupby(
        ["user", "session_id"],
        sort=True,
    ):
        session = session.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        times = session["timestamp"].to_numpy(dtype=np.int64)

        segment_ids = np.zeros(
            len(session),
            dtype=np.int32,
        )

        if len(session) > 1:
            segment_ids[1:] = np.cumsum(
                np.diff(times) > MOVE_CONTINUITY_GAP_MS
            )

        session["_segment"] = segment_ids
        count_session = 0

        for _, segment in session.groupby(
            "_segment",
            sort=False,
        ):
            result = segment_features(segment)
            if result is None:
                continue

            features, feature_times = result
            n_steps = len(features)

            for start in range(
                0,
                n_steps - SEQUENCE_LENGTH + 1,
                SEQUENCE_STRIDE,
            ):
                end = start + SEQUENCE_LENGTH

                sequences.append(
                    features[start:end]
                )

                meta_rows.append({
                    "sequence_index": sequence_index,
                    "user": str(user),
                    "session_id": str(session_id),
                    "sequence_start_ms": int(feature_times[start]),
                    "sequence_end_ms": int(feature_times[end - 1]),
                })

                sequence_index += 1
                count_session += 1

        print(
            f"   {str(user):<18} "
            f"{str(session_id):<20} "
            f"sequências={count_session:,}".replace(",", ".")
        )

    if not sequences:
        raise RuntimeError(
            "Nenhuma sequência foi gerada."
        )

    X = np.stack(
        sequences,
        axis=0,
    ).astype(np.float32, copy=False)

    metadata = pd.DataFrame(meta_rows)

    np.save(X_PATH, X, allow_pickle=False)

    metadata.to_csv(
        META_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    counts = (
        metadata.groupby("user")
        .agg(
            sequences=("sequence_index", "size"),
            sessions=("session_id", "nunique"),
        )
        .reset_index()
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(RAW_PATH),
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_stride": SEQUENCE_STRIDE,
        "features": FEATURE_NAMES,
        "n_sequences": int(len(metadata)),
        "shape": list(X.shape),
        "counts_by_user": counts.to_dict(orient="records"),
    }

    with CACHE_MANIFEST_PATH.open(
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
        f"\n   cache: {len(metadata):,} sequências".replace(",", ".")
    )

    print(counts.to_string(index=False))

    del sequences, meta_rows, X, metadata, move
    gc.collect()


def ensure_cache():
    if (
        X_PATH.exists()
        and META_PATH.exists()
        and CACHE_MANIFEST_PATH.exists()
    ):
        try:
            with CACHE_MANIFEST_PATH.open(
                "r",
                encoding="utf-8",
            ) as f:
                manifest = json.load(f)

            compatible = (
                manifest.get("sequence_length") == SEQUENCE_LENGTH
                and manifest.get("sequence_stride") == SEQUENCE_STRIDE
                and manifest.get("features") == FEATURE_NAMES
            )

            if compatible:
                print(
                    "Cache existente compatível. Reutilizando."
                )
                return
        except Exception:
            pass

        print("Cache incompatível. Recriando...")

    create_sequence_cache()


# ============================================================
# FOLDS POR SESSION_ID
# ============================================================

def build_folds(metadata: pd.DataFrame):
    users = sorted(
        metadata["user"].astype(str).unique().tolist()
    )

    fold_sessions = [
        []
        for _ in range(N_FOLDS)
    ]

    for user in users:
        sessions = sorted(
            metadata.loc[
                metadata["user"].astype(str).eq(user),
                "session_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        if len(sessions) < N_FOLDS:
            raise RuntimeError(
                f"{user}: apenas {len(sessions)} sessões."
            )

        rng = np.random.default_rng(
            stable_seed(f"folds|{user}")
        )

        rng.shuffle(sessions)

        for i, session in enumerate(sessions):
            fold_sessions[i % N_FOLDS].append(session)

    all_sessions = set(
        metadata["session_id"].astype(str).unique()
    )

    folds = []

    for i in range(N_FOLDS):
        test_sessions = sorted(set(fold_sessions[i]))

        folds.append({
            "fold": i + 1,
            "test_sessions": test_sessions,
            "train_sessions": sorted(
                all_sessions - set(test_sessions)
            ),
        })

    return folds


# ============================================================
# AMOSTRAGEM
# ============================================================

def sample_impostors(
    metadata: pd.DataFrame,
    genuine_indices: np.ndarray,
    impostor_pool_indices: np.ndarray,
    protected_user: str,
    fold_number: int,
):
    target = int(
        round(
            len(genuine_indices)
            * IMPOSTOR_TO_GENUINE_TRAIN_RATIO
        )
    )

    if len(impostor_pool_indices) <= target:
        return impostor_pool_indices

    pool = metadata.loc[
        impostor_pool_indices,
        ["user"],
    ].copy()

    pool["_idx"] = impostor_pool_indices

    users = sorted(
        pool["user"].astype(str).unique()
    )

    sampled_parts = []
    remaining = target

    for position, user in enumerate(users):
        user_indices = pool.loc[
            pool["user"].astype(str).eq(user),
            "_idx",
        ].to_numpy(dtype=np.int64)

        users_left = len(users) - position

        take = min(
            len(user_indices),
            max(1, remaining // users_left),
        )

        rng = np.random.default_rng(
            stable_seed(
                f"imp|{protected_user}|fold{fold_number}|{user}"
            )
        )

        sampled_parts.append(
            rng.choice(
                user_indices,
                size=take,
                replace=False,
            )
        )

        remaining -= take

    return np.concatenate(
        sampled_parts
    ).astype(np.int64)


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def scaler_from_train(
    X,
    train_indices: np.ndarray,
):
    # Neste dataset o recorte por fold cabe confortavelmente em RAM.
    # Ainda assim, calculamos em blocos.
    total = np.zeros(
        len(FEATURE_NAMES),
        dtype=np.float64,
    )

    total_sq = np.zeros(
        len(FEATURE_NAMES),
        dtype=np.float64,
    )

    count = 0

    for start in range(
        0,
        len(train_indices),
        1024,
    ):
        idx = train_indices[start:start + 1024]

        block = np.asarray(
            X[idx],
            dtype=np.float32,
        )

        total += block.sum(
            axis=(0, 1),
            dtype=np.float64,
        )

        total_sq += (
            block.astype(np.float64) ** 2
        ).sum(axis=(0, 1))

        count += block.shape[0] * block.shape[1]

        del block

    mean = total / max(count, 1)

    variance = (
        total_sq / max(count, 1)
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        1e-8,
    )

    std = np.sqrt(variance)

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


# ============================================================
# MODELO TENSORFLOW
# ============================================================

def build_model(
    input_shape,
):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=input_shape
        ),
        tf.keras.layers.LSTM(
            HIDDEN_SIZE
        ),
        tf.keras.layers.Dropout(
            DROPOUT
        ),
        tf.keras.layers.Dense(
            DENSE_SIZE,
            activation="relu",
        ),
        tf.keras.layers.Dropout(
            DROPOUT
        ),
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
    )

    return model


# ============================================================
# MÉTRICAS
# ============================================================

def compute_eer(
    y_true,
    scores,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    idx = int(
        np.nanargmin(
            np.abs(fpr - fnr)
        )
    )

    far = float(fpr[idx])
    frr = float(fnr[idx])

    return {
        "eer": float((far + frr) / 2.0),
        "eer_threshold": float(thresholds[idx]),
        "far_at_eer": far,
        "frr_at_eer": frr,
    }


def auth_metrics(
    y_true,
    scores,
):
    y_true = np.asarray(
        y_true,
        dtype=np.int8,
    )

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    if len(np.unique(y_true)) < 2:
        raise RuntimeError(
            "Teste sem genuínos e impostores."
        )

    eer_info = compute_eer(
        y_true,
        scores,
    )

    pred = (
        scores
        >= eer_info["eer_threshold"]
    ).astype(np.int8)

    return {
        "n_test": int(len(y_true)),
        "n_genuine": int(np.sum(y_true == 1)),
        "n_impostor": int(np.sum(y_true == 0)),
        **eer_info,
        "roc_auc": float(
            roc_auc_score(y_true, scores)
        ),
        "f1_genuine_eer": float(
            f1_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "balanced_accuracy_eer": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
        "accuracy_eer": float(
            accuracy_score(
                y_true,
                pred,
            )
        ),
        "precision_genuine_eer": float(
            precision_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_genuine_eer": float(
            recall_score(
                y_true,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
    }


# ============================================================
# AGREGAÇÃO TEMPORAL
# ============================================================

def aggregate_one_session(
    session: pd.DataFrame,
    horizon_s: int,
):
    g = session.sort_values(
        [
            "sequence_end_ms",
            "sequence_start_ms",
            "sequence_index",
        ]
    ).reset_index(drop=True)

    times = g["sequence_end_ms"].to_numpy(dtype=np.int64)
    scores = g["score_genuine"].to_numpy(dtype=np.float32)
    labels = g["y_true_genuine"].to_numpy(dtype=np.int8)

    if horizon_s == 0:
        return pd.DataFrame({
            "y_true_genuine": labels,
            "score_eval": scores,
            "n_scores": np.ones(
                len(g),
                dtype=np.int16,
            ),
        })

    horizon_ms = horizon_s * 1000
    cumulative = np.concatenate(
        [[0.0], np.cumsum(scores, dtype=np.float64)]
    )

    first_time = int(times[0])
    left = 0

    out_y = []
    out_score = []
    out_n = []

    for i in range(len(g)):
        now = int(times[i])

        while (
            left < i
            and now - int(times[left]) > horizon_ms
        ):
            left += 1

        n = i - left + 1

        if n < TEMPORAL_MIN_SCORES:
            continue

        if (
            TEMPORAL_REQUIRE_FULL_HORIZON
            and now - first_time < horizon_ms
        ):
            continue

        mean_score = (
            cumulative[i + 1] - cumulative[left]
        ) / n

        out_y.append(int(labels[i]))
        out_score.append(float(mean_score))
        out_n.append(int(n))

    return pd.DataFrame({
        "y_true_genuine": out_y,
        "score_eval": out_score,
        "n_scores": out_n,
    })


def temporal_results(
    predictions: pd.DataFrame,
):
    rows = []

    for (
        protected_user,
        fold,
    ), fold_df in predictions.groupby(
        ["protected_user", "fold"],
        sort=True,
    ):
        for horizon_s in TEMPORAL_HORIZONS_SECONDS:
            parts = []

            for _, session in fold_df.groupby(
                "session_id",
                sort=False,
            ):
                part = aggregate_one_session(
                    session,
                    horizon_s,
                )

                if not part.empty:
                    parts.append(part)

            if not parts:
                continue

            decisions = pd.concat(
                parts,
                ignore_index=True,
            )

            m = auth_metrics(
                decisions["y_true_genuine"].to_numpy(),
                decisions["score_eval"].to_numpy(),
            )

            rows.append({
                "protected_user": str(protected_user),
                "fold": int(fold),
                "horizon_s": int(horizon_s),
                "mean_scores_per_decision": float(
                    decisions["n_scores"].mean()
                ),
                **m,
            })

            del decisions, parts
            gc.collect()

    return pd.DataFrame(rows)


# ============================================================
# RESUMOS
# ============================================================

def summarize(
    df: pd.DataFrame,
    group_cols: list[str],
):
    metrics = [
        "eer",
        "far_at_eer",
        "frr_at_eer",
        "roc_auc",
        "f1_genuine_eer",
        "balanced_accuracy_eer",
        "accuracy_eer",
        "precision_genuine_eer",
        "recall_genuine_eer",
    ]

    rows = []

    for keys, group in df.groupby(
        group_cols,
        sort=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            col: value
            for col, value in zip(
                group_cols,
                keys,
            )
        }

        row["n_folds"] = len(group)

        for metric in metrics:
            if metric not in group.columns:
                continue

            v = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            row[f"{metric}_mean"] = float(v.mean())
            row[f"{metric}_std"] = float(v.std(ddof=0))

        if "mean_scores_per_decision" in group.columns:
            row["mean_scores_per_decision_mean"] = float(
                group["mean_scores_per_decision"].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def choose_best_temporal(
    summary: pd.DataFrame,
):
    rows = []

    for user, group in summary.groupby(
        "protected_user",
        sort=True,
    ):
        ranked = group.sort_values(
            [
                "eer_mean",
                "roc_auc_mean",
                "f1_genuine_eer_mean",
                "horizon_s",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )

        rows.append(
            ranked.iloc[0].copy()
        )

    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 112)
    print(
        "CYBELL-J — FASE 12 — "
        "TREINAMENTO LSTM TENSORFLOW/KERAS"
    )
    print("=" * 112)

    configure_tensorflow()

    print(f"\nTensorFlow: {tf.__version__}")
    print("Backend: TensorFlow / Keras")
    print("Dispositivo: CPU (Windows nativo)")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"LSTM hidden: {HIDDEN_SIZE}")
    print(f"Máximo de épocas: {MAX_EPOCHS}")

    physical_gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visíveis ao TensorFlow: {physical_gpus}")

    seed_everything(RANDOM_STATE)
    reset_outputs()

    print("\n1) Preparando sequências...")
    ensure_cache()

    X = np.load(
        X_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    metadata = pd.read_csv(
        META_PATH,
        low_memory=False,
        dtype={
            "sequence_index": "int64",
            "user": "string",
            "session_id": "string",
            "sequence_start_ms": "int64",
            "sequence_end_ms": "int64",
        },
    ).sort_values(
        "sequence_index"
    ).reset_index(drop=True)

    if len(metadata) != X.shape[0]:
        raise RuntimeError(
            "X e metadata têm tamanhos diferentes."
        )

    print(f"   X shape: {X.shape}")

    counts = (
        metadata.groupby("user")
        .agg(
            sequences=("sequence_index", "size"),
            sessions=("session_id", "nunique"),
        )
        .reset_index()
    )

    print(counts.to_string(index=False))

    users = sorted(
        metadata["user"].astype(str).unique()
    )

    folds = build_folds(metadata)

    print("\n2) Folds externos por session_id:")
    for fd in folds:
        print(
            f"   Fold {fd['fold']}: "
            f"{len(fd['test_sessions'])} sessões de teste"
        )

    print("\n" + "=" * 112)
    print("3) TREINANDO — 6 PERFIS x 3 FOLDS")
    print("=" * 112)

    for user_pos, protected_user in enumerate(
        users,
        start=1,
    ):
        print("\n" + "-" * 112)
        print(
            f"[{user_pos}/{len(users)}] "
            f"IDENTIDADE PROTEGIDA: {protected_user}"
        )
        print("-" * 112)

        for fold_info in folds:
            fold = int(fold_info["fold"])
            test_sessions = set(
                fold_info["test_sessions"]
            )

            train_mask = ~metadata[
                "session_id"
            ].astype(str).isin(test_sessions)

            test_mask = metadata[
                "session_id"
            ].astype(str).isin(test_sessions)

            genuine_mask = (
                train_mask
                & metadata["user"]
                .astype(str)
                .eq(protected_user)
            )

            impostor_mask = (
                train_mask
                & ~metadata["user"]
                .astype(str)
                .eq(protected_user)
            )

            genuine_idx = np.flatnonzero(
                genuine_mask.to_numpy()
            ).astype(np.int64)

            impostor_pool_idx = np.flatnonzero(
                impostor_mask.to_numpy()
            ).astype(np.int64)

            test_idx = np.flatnonzero(
                test_mask.to_numpy()
            ).astype(np.int64)

            sampled_imp = sample_impostors(
                metadata,
                genuine_idx,
                impostor_pool_idx,
                protected_user,
                fold,
            )

            train_idx = np.concatenate([
                genuine_idx,
                sampled_imp,
            ])

            rng = np.random.default_rng(
                stable_seed(
                    f"shuffle|{protected_user}|{fold}"
                )
            )
            rng.shuffle(train_idx)

            y_train = (
                metadata.loc[
                    train_idx,
                    "user",
                ]
                .astype(str)
                .eq(protected_user)
                .astype(np.float32)
                .to_numpy()
            )

            y_test = (
                metadata.loc[
                    test_idx,
                    "user",
                ]
                .astype(str)
                .eq(protected_user)
                .astype(np.int8)
                .to_numpy()
            )

            print(
                f"\n   Fold {fold} | "
                f"treino={len(train_idx):,} "
                f"(genuíno={len(genuine_idx):,}, "
                f"impostor={len(sampled_imp):,}) | "
                f"teste={len(test_idx):,}".replace(",", ".")
            )

            mean, std = scaler_from_train(
                X,
                train_idx,
            )

            # Cópias por fold são relativamente pequenas e evitam
            # pipelines complexos, mantendo a execução estável.
            X_train = np.asarray(
                X[train_idx],
                dtype=np.float32,
            )

            X_test = np.asarray(
                X[test_idx],
                dtype=np.float32,
            )

            X_train = (
                X_train - mean
            ) / std

            X_test = (
                X_test - mean
            ) / std

            run_seed = stable_seed(
                f"tf|{protected_user}|fold{fold}"
            )

            seed_everything(run_seed)

            tf.keras.backend.clear_session()

            model = build_model(
                input_shape=(
                    SEQUENCE_LENGTH,
                    len(FEATURE_NAMES),
                )
            )

            class_weight = {
                0: 1.0,
                1: float(
                    len(sampled_imp)
                    / max(len(genuine_idx), 1)
                ),
            }

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=EARLY_STOPPING_PATIENCE,
                    min_delta=MIN_DELTA,
                    restore_best_weights=True,
                    verbose=1,
                )
            ]

            start = datetime.now()

            history = model.fit(
                X_train,
                y_train,
                epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=2,
                class_weight=class_weight,
                callbacks=callbacks,
            )

            fit_seconds = (
                datetime.now() - start
            ).total_seconds()

            scores = (
                model.predict(
                    X_test,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                )
                .reshape(-1)
                .astype(np.float32)
            )

            m = auth_metrics(
                y_test,
                scores,
            )

            print(
                f"   RESULTADO | "
                f"EER={m['eer']:.4f} | "
                f"AUC={m['roc_auc']:.4f} | "
                f"F1={m['f1_genuine_eer']:.4f} | "
                f"BalAcc={m['balanced_accuracy_eer']:.4f}"
            )

            row = {
                "protected_user": protected_user,
                "fold": fold,
                "n_train": len(train_idx),
                "n_train_genuine": len(genuine_idx),
                "n_train_impostor": len(sampled_imp),
                "epochs_ran": len(history.history["loss"]),
                "best_train_loss": float(
                    np.min(history.history["loss"])
                ),
                "fit_seconds": fit_seconds,
                "framework": "TensorFlow/Keras",
                "device": "CPU",
                **m,
            }

            append_csv(
                pd.DataFrame([row]),
                FOLD_RESULTS_PATH,
            )

            pred_meta = metadata.loc[
                test_idx
            ].copy()

            pred_df = pd.DataFrame({
                "protected_user": protected_user,
                "fold": fold,
                "sequence_index": test_idx,
                "session_id": pred_meta[
                    "session_id"
                ].astype(str).to_numpy(),
                "actual_user": pred_meta[
                    "user"
                ].astype(str).to_numpy(),
                "sequence_start_ms": pred_meta[
                    "sequence_start_ms"
                ].to_numpy(dtype=np.int64),
                "sequence_end_ms": pred_meta[
                    "sequence_end_ms"
                ].to_numpy(dtype=np.int64),
                "y_true_genuine": y_test,
                "score_genuine": scores,
            })

            append_csv(
                pred_df,
                PREDICTIONS_PATH,
            )

            model_path = (
                MODELS_DIR
                / f"{protected_user}_fold{fold}.keras"
            )

            model.save(
                model_path,
                include_optimizer=False,
            )

            scaler_path = (
                MODELS_DIR
                / f"{protected_user}_fold{fold}_scaler.npz"
            )

            np.savez(
                scaler_path,
                mean=mean,
                std=std,
                feature_names=np.array(
                    FEATURE_NAMES
                ),
            )

            del (
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                scores,
                pred_df,
                pred_meta,
                train_idx,
                genuine_idx,
                sampled_imp,
                impostor_pool_idx,
                test_idx,
                mean,
                std,
                history,
            )

            tf.keras.backend.clear_session()
            gc.collect()

            print("   memória temporária liberada.")

    # ========================================================
    # RESUMO SINGLE
    # ========================================================

    print("\n" + "=" * 112)
    print("4) RESUMO LSTM — JANELA ÚNICA")
    print("=" * 112)

    fold_results = pd.read_csv(
        FOLD_RESULTS_PATH,
        low_memory=False,
    )

    single_summary = summarize(
        fold_results,
        ["protected_user"],
    )

    single_summary.to_csv(
        SINGLE_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        single_summary[
            [
                "protected_user",
                "eer_mean",
                "roc_auc_mean",
                "f1_genuine_eer_mean",
                "balanced_accuracy_eer_mean",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # TEMPORAL
    # ========================================================

    print("\n" + "=" * 112)
    print("5) AGREGAÇÃO TEMPORAL LSTM")
    print("=" * 112)

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        low_memory=False,
    )

    temporal_folds = temporal_results(
        predictions
    )

    temporal_folds.to_csv(
        TEMPORAL_FOLDS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    temporal_summary = summarize(
        temporal_folds,
        [
            "protected_user",
            "horizon_s",
        ],
    )

    temporal_summary.to_csv(
        TEMPORAL_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    best_temporal = choose_best_temporal(
        temporal_summary
    )

    best_temporal.to_csv(
        BEST_TEMPORAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nMelhor horizonte por perfil:"
    )

    print(
        best_temporal[
            [
                "protected_user",
                "horizon_s",
                "eer_mean",
                "roc_auc_mean",
                "f1_genuine_eer_mean",
                "balanced_accuracy_eer_mean",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # COMPARAÇÃO COM CLÁSSICO / PYCARET
    # ========================================================

    comparison = pd.DataFrame()

    if CLASSIC_BASELINE_PATH.exists():
        classic = pd.read_csv(
            CLASSIC_BASELINE_PATH,
            low_memory=False,
        )

        if "protected_user" in classic.columns:
            classic_cols = [
                c for c in [
                    "protected_user",
                    "candidate",
                    "strategy",
                    "horizon_s",
                    "eer_mean",
                    "roc_auc_mean",
                    "f1_genuine_eer_mean",
                    "balanced_accuracy_eer_mean",
                ]
                if c in classic.columns
            ]

            classic_small = classic[
                classic_cols
            ].copy()

            classic_small = classic_small.rename(
                columns={
                    c: (
                        c
                        if c == "protected_user"
                        else f"classic_{c}"
                    )
                    for c in classic_small.columns
                }
            )

            lstm_small = best_temporal[
                [
                    "protected_user",
                    "horizon_s",
                    "eer_mean",
                    "roc_auc_mean",
                    "f1_genuine_eer_mean",
                    "balanced_accuracy_eer_mean",
                ]
            ].rename(
                columns={
                    "horizon_s": "lstm_horizon_s",
                    "eer_mean": "lstm_eer_mean",
                    "roc_auc_mean": "lstm_roc_auc_mean",
                    "f1_genuine_eer_mean": (
                        "lstm_f1_genuine_eer_mean"
                    ),
                    "balanced_accuracy_eer_mean": (
                        "lstm_balanced_accuracy_eer_mean"
                    ),
                }
            )

            comparison = lstm_small.merge(
                classic_small,
                on="protected_user",
                how="left",
            )

            if "classic_eer_mean" in comparison.columns:
                comparison[
                    "eer_delta_lstm_minus_classic"
                ] = (
                    comparison["lstm_eer_mean"]
                    - comparison["classic_eer_mean"]
                )

                comparison[
                    "winner_lower_eer"
                ] = np.where(
                    comparison["lstm_eer_mean"]
                    < comparison["classic_eer_mean"],
                    "LSTM_TENSORFLOW",
                    "CLASSICO_PYCARET",
                )

            comparison.to_csv(
                COMPARISON_PATH,
                index=False,
                encoding="utf-8-sig",
            )

            show = [
                c for c in [
                    "protected_user",
                    "lstm_horizon_s",
                    "lstm_eer_mean",
                    "classic_horizon_s",
                    "classic_eer_mean",
                    "eer_delta_lstm_minus_classic",
                    "winner_lower_eer",
                ]
                if c in comparison.columns
            ]

            print(
                "\nComparação LSTM TensorFlow x clássico/PyCaret:"
            )

            print(
                comparison[show].to_string(
                    index=False
                )
            )
    else:
        print(
            "\nBaseline clássico não encontrado; "
            "comparação automática pulada."
        )

    # ========================================================
    # GLOBAL
    # ========================================================

    global_row = {
        "model": "LSTM_TENSORFLOW",
        "framework": "TensorFlow/Keras",
        "device": "CPU",
        "profiles": int(
            best_temporal["protected_user"].nunique()
        ),
        "eer_mean_profiles": float(
            best_temporal["eer_mean"].mean()
        ),
        "eer_median_profiles": float(
            best_temporal["eer_mean"].median()
        ),
        "auc_mean_profiles": float(
            best_temporal["roc_auc_mean"].mean()
        ),
        "f1_eer_mean_profiles": float(
            best_temporal["f1_genuine_eer_mean"].mean()
        ),
        "balanced_accuracy_eer_mean_profiles": float(
            best_temporal[
                "balanced_accuracy_eer_mean"
            ].mean()
        ),
    }

    global_df = pd.DataFrame([global_row])

    global_df.to_csv(
        GLOBAL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 112)
    print("6) RESULTADO GLOBAL LSTM TENSORFLOW")
    print("=" * 112)

    print(global_df.to_string(index=False))

    manifest = {
        "projeto": "CYBELL-J",
        "fase": (
            "12 - treinamento LSTM TensorFlow/Keras "
            "autenticacao continua 1:1"
        ),
        "gerado_em_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "tensorflow_version": tf.__version__,
        "framework": "TensorFlow/Keras",
        "device": "CPU - Windows native",
        "cpu_threads": CPU_THREADS,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_stride": SEQUENCE_STRIDE,
        "features": FEATURE_NAMES,
        "model": {
            "lstm_units": HIDDEN_SIZE,
            "dense_units": DENSE_SIZE,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        "validation": {
            "folds": N_FOLDS,
            "split": "session_id",
            "random_window_split": False,
            "impostor_to_genuine_ratio": (
                IMPOSTOR_TO_GENUINE_TRAIN_RATIO
            ),
        },
        "pycaret_note": (
            "PyCaret permanece como baseline clássico. "
            "A LSTM TensorFlow é comparada aos resultados clássicos "
            "salvos pelo pipeline anterior."
        ),
        "global_result": global_row,
        "methodological_notes": [
            "TensorFlow moderno em Windows nativo é usado em CPU.",
            "A LSTM usa sequências evento-a-evento e não as 89 métricas agregadas.",
            "Normalização é calculada somente com treino de cada fold.",
            "Nenhuma session_id é compartilhada entre treino e teste.",
            "Early stopping monitora loss de treino para não usar a sessão externa de teste.",
            "Threshold de EER do teste é apenas métrica de pesquisa, não threshold de produção.",
        ],
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

    del (
        predictions,
        temporal_folds,
        temporal_summary,
        best_temporal,
        fold_results,
        single_summary,
        metadata,
        X,
    )

    tf.keras.backend.clear_session()
    gc.collect()

    print("\n" + "=" * 112)
    print("FASE 12 TENSORFLOW CONCLUÍDA")
    print("=" * 112)

    print(f"\nResultados por fold:\n{FOLD_RESULTS_PATH}")
    print(f"\nPredições OOF:\n{PREDICTIONS_PATH}")
    print(f"\nMelhor horizonte:\n{BEST_TEMPORAL_PATH}")
    print(f"\nComparação com clássico/PyCaret:\n{COMPARISON_PATH}")
    print(f"\nResultado global:\n{GLOBAL_PATH}")
    print(f"\nManifesto:\n{MANIFEST_PATH}")


if __name__ == "__main__":
    main()