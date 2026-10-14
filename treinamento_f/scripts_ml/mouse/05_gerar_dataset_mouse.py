"""
05_gerar_dataset_mouse.py

Gera o dataset de janelas comportamentais da parte de mouse/trackpad do CYBELL-J.

Entrada:
    data/mouse_snapshot/mouse_events_raw.csv
    data/mouse_snapshot/mouse_events_metadata.json

Saídas:
    data/mouse_dataset/dataset_mouse.csv
    data/mouse_dataset/dataset_device_high_confidence.csv
    data/mouse_dataset/resumo_dataset.csv
    data/mouse_dataset/dataset_manifest.json

Princípios:
- NÃO acessa nem altera o MongoDB.
- NÃO altera o snapshot RAW.
- Verifica o SHA-256 do snapshot antes de processar.
- Reconstrói sessões por inatividade > 300 s.
- Gera janelas de 25, 30, 50 e 100 eventos.
- Gera sobreposições solicitadas de 25%, 50% e 75%.
- Calcula cinemática apenas sobre mouse_move.
- Calcula clique/pressionamento/scroll separadamente.
- Mantém mouse/trackpad como contexto/metadado.
- Cria uma base separada de contexto de dispositivo com apenas janelas
  de alta confiança (>=95%), lembrando que o rótulo vem da heurística
  do front-end e NÃO é ground truth físico de hardware.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

SNAPSHOT_DIR = BASE_DIR / "data" / "mouse_snapshot"
RAW_CSV = SNAPSHOT_DIR / "mouse_events_raw.csv"
RAW_METADATA = SNAPSHOT_DIR / "mouse_events_metadata.json"

OUTPUT_DIR = BASE_DIR / "data" / "mouse_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = OUTPUT_DIR / "dataset_mouse.csv"
DEVICE_PATH = OUTPUT_DIR / "dataset_device_high_confidence.csv"
SUMMARY_PATH = OUTPUT_DIR / "resumo_dataset.csv"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.json"

SESSION_GAP_SECONDS = 300.0

WINDOW_SIZES = [25, 30, 50, 100]
REQUESTED_OVERLAPS = [0.25, 0.50, 0.75]

DEVICE_CONTEXT_THRESHOLD = 0.80
DEVICE_HIGH_CONFIDENCE_THRESHOLD = 0.95

DEVICE_LABEL_SOURCE = "front_end_heuristic"


# ============================================================
# UTILITÁRIOS
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def half_up(value: float) -> int:
    """Arredondamento .5 para cima, evitando o banker's rounding do Python."""
    return int(math.floor(value + 0.5))


def finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def stats_basic(values, prefix: str, include_min: bool = False) -> dict:
    arr = finite_array(values)

    result = {
        f"{prefix}_mean": 0.0,
        f"{prefix}_std": 0.0,
        f"{prefix}_median": 0.0,
        f"{prefix}_p90": 0.0,
        f"{prefix}_max": 0.0,
    }

    if include_min:
        result[f"{prefix}_min"] = 0.0

    if arr.size == 0:
        return result

    result[f"{prefix}_mean"] = float(np.mean(arr))
    result[f"{prefix}_std"] = float(np.std(arr, ddof=0))
    result[f"{prefix}_median"] = float(np.median(arr))
    result[f"{prefix}_p90"] = float(np.percentile(arr, 90))
    result[f"{prefix}_max"] = float(np.max(arr))

    if include_min:
        result[f"{prefix}_min"] = float(np.min(arr))

    return result


def safe_rate(count: int, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return float(count / duration_s)


def normalize_turn_angles(angle_rad: np.ndarray) -> np.ndarray:
    """
    Diferença angular absoluta entre vetores consecutivos,
    normalizada para [0, pi].
    """
    if angle_rad.size < 2:
        return np.array([], dtype=float)

    delta = np.diff(angle_rad)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return np.abs(delta)


def pair_hold_times(event_types: np.ndarray, timestamps_ms: np.ndarray) -> np.ndarray:
    """
    Pareia genericamente mouse_down -> próximo mouse_up.
    O banco não preservou o botão, portanto NÃO identifica botão esquerdo/direito.
    """
    holds = []
    down_time = None

    for event_type, ts in zip(event_types, timestamps_ms):
        if event_type == "mouse_down":
            down_time = ts
        elif event_type == "mouse_up" and down_time is not None:
            hold = ts - down_time
            if hold >= 0:
                holds.append(hold)
            down_time = None

    return np.asarray(holds, dtype=float)


def classify_device(device_values: np.ndarray) -> tuple:
    valid = [str(v).strip().lower() for v in device_values if pd.notna(v)]

    total = len(valid)

    if total == 0:
        return 0.0, 0.0, "unknown", 0.0, 0

    mouse_count = sum(v == "mouse" for v in valid)
    trackpad_count = sum(v == "trackpad" for v in valid)

    mouse_pct = mouse_count / total
    trackpad_pct = trackpad_count / total

    confidence = max(mouse_pct, trackpad_pct)

    if mouse_pct >= DEVICE_CONTEXT_THRESHOLD:
        context = "mouse"
    elif trackpad_pct >= DEVICE_CONTEXT_THRESHOLD:
        context = "trackpad"
    else:
        context = "mixed"

    high_conf = int(
        context in ("mouse", "trackpad")
        and confidence >= DEVICE_HIGH_CONFIDENCE_THRESHOLD
    )

    return mouse_pct, trackpad_pct, context, confidence, high_conf


# ============================================================
# EXTRAÇÃO DE FEATURES DE UMA JANELA
# ============================================================

def extract_window_features(
    user: str,
    session_id: str,
    window_number: int,
    window_size: int,
    requested_overlap: float,
    step_events: int,
    effective_overlap: float,
    event_types: np.ndarray,
    device_types: np.ndarray,
    timestamps_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    delta_x: np.ndarray,
    delta_y: np.ndarray,
) -> dict:

    n_events = len(event_types)

    start_ms = float(timestamps_ms[0])
    end_ms = float(timestamps_ms[-1])
    duration_s = max(0.0, (end_ms - start_ms) / 1000.0)

    # --------------------------------------------------------
    # Metadados do dispositivo
    # --------------------------------------------------------

    (
        mouse_pct,
        trackpad_pct,
        device_context,
        device_confidence,
        device_high_confidence,
    ) = classify_device(device_types)

    # --------------------------------------------------------
    # Tipos de eventos
    # --------------------------------------------------------

    move_mask = event_types == "mouse_move"
    click_mask = event_types == "mouse_click"
    down_mask = event_types == "mouse_down"
    up_mask = event_types == "mouse_up"
    wheel_mask = event_types == "mouse_wheel"

    n_move = int(np.sum(move_mask))
    n_click = int(np.sum(click_mask))
    n_down = int(np.sum(down_mask))
    n_up = int(np.sum(up_mask))
    n_wheel = int(np.sum(wheel_mask))

    # --------------------------------------------------------
    # Intervalo entre todos os eventos
    # --------------------------------------------------------

    event_dt_s = np.diff(timestamps_ms) / 1000.0
    event_dt_s = event_dt_s[event_dt_s > 0]

    # --------------------------------------------------------
    # Cinemática: SOMENTE mouse_move
    # --------------------------------------------------------

    move_t = timestamps_ms[move_mask]
    move_x = x[move_mask]
    move_y = y[move_mask]

    valid_move = (
        np.isfinite(move_t)
        & np.isfinite(move_x)
        & np.isfinite(move_y)
    )

    move_t = move_t[valid_move]
    move_x = move_x[valid_move]
    move_y = move_y[valid_move]

    dx_move = np.diff(move_x)
    dy_move = np.diff(move_y)
    dt_move_s = np.diff(move_t) / 1000.0

    positive_dt = dt_move_s > 0

    dx_valid = dx_move[positive_dt]
    dy_valid = dy_move[positive_dt]
    dt_valid = dt_move_s[positive_dt]

    distance = np.sqrt(dx_valid**2 + dy_valid**2)

    if distance.size:
        speed = distance / dt_valid
    else:
        speed = np.array([], dtype=float)

    # A aceleração usa diferenças entre velocidades consecutivas.
    if speed.size >= 2 and dt_valid.size >= 2:
        accel_dt = dt_valid[1:]
        positive_accel_dt = accel_dt > 0

        acceleration = np.diff(speed)[positive_accel_dt] / accel_dt[positive_accel_dt]
    else:
        acceleration = np.array([], dtype=float)

    abs_acceleration = np.abs(acceleration)

    # Ângulo e mudança angular
    if dx_valid.size:
        angle_rad = np.arctan2(dy_valid, dx_valid)
        turn_abs_deg = np.degrees(normalize_turn_angles(angle_rad))
    else:
        angle_rad = np.array([], dtype=float)
        turn_abs_deg = np.array([], dtype=float)

    total_distance = float(np.sum(distance)) if distance.size else 0.0

    if move_x.size >= 2:
        displacement = float(
            np.sqrt(
                (move_x[-1] - move_x[0]) ** 2
                + (move_y[-1] - move_y[0]) ** 2
            )
        )
    else:
        displacement = 0.0

    straightness = (
        displacement / total_distance
        if total_distance > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Clique / pressionamento
    # --------------------------------------------------------

    click_times = timestamps_ms[click_mask]
    click_intervals_s = np.diff(click_times) / 1000.0
    click_intervals_s = click_intervals_s[click_intervals_s > 0]

    hold_times_ms = pair_hold_times(
        event_types,
        timestamps_ms
    )

    # --------------------------------------------------------
    # Scroll
    # --------------------------------------------------------

    wheel_dx = delta_x[wheel_mask]
    wheel_dy = delta_y[wheel_mask]

    wheel_dx = finite_array(wheel_dx)
    wheel_dy = finite_array(wheel_dy)

    abs_wheel_dx = np.abs(wheel_dx)
    abs_wheel_dy = np.abs(wheel_dy)

    if n_wheel:
        wheel_horizontal_fraction = float(
            np.mean(abs_wheel_dx > 0)
        ) if abs_wheel_dx.size else 0.0
    else:
        wheel_horizontal_fraction = 0.0

    if wheel_dy.size >= 2:
        sign = np.sign(wheel_dy)
        nonzero = sign != 0
        sign = sign[nonzero]
        if sign.size >= 2:
            wheel_direction_changes = int(
                np.sum(sign[1:] != sign[:-1])
            )
        else:
            wheel_direction_changes = 0
    else:
        wheel_direction_changes = 0

    # --------------------------------------------------------
    # Linha final
    # --------------------------------------------------------

    row = {
        # alvo / identificação
        "target_user": user,
        "user": user,
        "session_id": session_id,
        "window_id": f"{session_id}_W{window_size}_O{int(requested_overlap*100):02d}_{window_number:06d}",

        # configuração experimental
        "window_size": int(window_size),
        "requested_overlap_pct": int(round(requested_overlap * 100)),
        "step_events": int(step_events),
        "effective_overlap_pct": round(effective_overlap * 100, 4),

        # tempo
        "window_start_ms": int(start_ms),
        "window_end_ms": int(end_ms),
        "duration_s": float(duration_s),

        # contagens
        "n_events": int(n_events),
        "n_mouse_move": n_move,
        "n_mouse_click": n_click,
        "n_mouse_down": n_down,
        "n_mouse_up": n_up,
        "n_mouse_wheel": n_wheel,

        # contexto do dispositivo - METADADO
        "device_mouse_pct": float(mouse_pct * 100),
        "device_trackpad_pct": float(trackpad_pct * 100),
        "device_context": device_context,
        "device_confidence": float(device_confidence),
        "device_high_confidence": int(device_high_confidence),
        "device_label_source": DEVICE_LABEL_SOURCE,

        # alvo de subperfil para experimento condicionado
        "target_user_device": f"{user}__{device_context}",

        # taxas
        "event_rate_hz": safe_rate(n_events, duration_s),
        "move_rate_hz": safe_rate(n_move, duration_s),
        "click_rate_hz": safe_rate(n_click, duration_s),
        "wheel_rate_hz": safe_rate(n_wheel, duration_s),

        # trajetória
        "movement_steps_valid": int(distance.size),
        "distance_total_px": total_distance,
        "displacement_px": displacement,
        "straightness": float(straightness),

        # scroll complementar
        "wheel_horizontal_fraction": wheel_horizontal_fraction,
        "wheel_direction_changes": wheel_direction_changes,

        # holds
        "hold_count": int(hold_times_ms.size),
    }

    row.update(stats_basic(event_dt_s, "event_dt_s"))
    row.update(stats_basic(dt_valid, "move_dt_s"))
    row.update(stats_basic(distance, "distance_px"))
    row.update(stats_basic(speed, "speed_px_s"))
    row.update(stats_basic(acceleration, "accel_px_s2", include_min=True))
    row.update(stats_basic(abs_acceleration, "abs_accel_px_s2"))
    row.update(stats_basic(turn_abs_deg, "turn_abs_deg"))
    row.update(stats_basic(np.abs(dx_valid), "move_abs_dx_px"))
    row.update(stats_basic(np.abs(dy_valid), "move_abs_dy_px"))
    row.update(stats_basic(click_intervals_s, "click_interval_s"))
    row.update(stats_basic(hold_times_ms, "hold_ms"))
    row.update(stats_basic(abs_wheel_dx, "wheel_abs_dx"))
    row.update(stats_basic(abs_wheel_dy, "wheel_abs_dy"))
    row.update(stats_basic(wheel_dy, "wheel_dy", include_min=True))

    return row


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def prepare_dataframe() -> pd.DataFrame:
    print("=" * 78)
    print("GERAÇÃO DO DATASET MOUSE/TRACKPAD — CYBELL-J")
    print("=" * 78)

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Snapshot não encontrado:\n{RAW_CSV}"
        )

    if not RAW_METADATA.exists():
        raise FileNotFoundError(
            f"Metadados do snapshot não encontrados:\n{RAW_METADATA}"
        )

    with RAW_METADATA.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected_hash = metadata.get("sha256")

    print("\n1) Verificando integridade do snapshot...")
    current_hash = sha256_file(RAW_CSV)

    print(f"   SHA-256 esperado: {expected_hash}")
    print(f"   SHA-256 atual:    {current_hash}")

    if expected_hash and current_hash != expected_hash:
        raise RuntimeError(
            "O SHA-256 do snapshot não confere. "
            "O arquivo RAW pode ter sido alterado."
        )

    print("   ✅ Snapshot íntegro.")

    print("\n2) Carregando dados...")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    required = [
        "_id",
        "user",
        "timestamp",
        "event_type",
        "device_type",
        "x",
        "y",
        "delta_x",
        "delta_y",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"Colunas obrigatórias ausentes: {missing}"
        )

    print(f"   ✅ {len(df)} eventos carregados.")

    # Numéricos
    for col in ["timestamp", "x", "y", "delta_x", "delta_y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["_id", "user", "timestamp", "event_type", "device_type"]
    ).copy()

    df["user"] = df["user"].astype(str).str.strip()
    df["event_type"] = df["event_type"].astype(str).str.strip()
    df["device_type"] = df["device_type"].astype(str).str.strip().str.lower()
    df["_id"] = df["_id"].astype(str)

    # Ordenação determinística:
    # timestamp e, em empates, _id.
    df = df.sort_values(
        ["user", "timestamp", "_id"]
    ).reset_index(drop=True)

    # Sessões por inatividade > 300 s
    df["gap_s"] = (
        df.groupby("user")["timestamp"].diff() / 1000.0
    )

    new_session = (
        df["gap_s"].isna()
        | (df["gap_s"] > SESSION_GAP_SECONDS)
    )

    df["session_number"] = (
        new_session
        .groupby(df["user"])
        .cumsum()
        .astype(int)
    )

    df["session_id"] = (
        df["user"]
        + "_S"
        + df["session_number"].astype(str).str.zfill(2)
    )

    print(
        f"   ✅ {df['session_id'].nunique()} sessões reconstruídas "
        f"(pausa > {int(SESSION_GAP_SECONDS)} s)."
    )

    return df


# ============================================================
# GERAÇÃO
# ============================================================

def main():
    df = prepare_dataframe()

    print("\n3) Gerando janelas e características...")
    print("   Configurações:")
    for ws in WINDOW_SIZES:
        for overlap in REQUESTED_OVERLAPS:
            step = max(1, half_up(ws * (1.0 - overlap)))
            effective = 1.0 - (step / ws)
            print(
                f"   - {ws:>3} eventos | overlap solicitado "
                f"{int(overlap*100):>2}% | passo {step:>2} | "
                f"overlap efetivo {effective*100:.2f}%"
            )

    # Prepara arrays por sessão apenas uma vez
    sessions = []

    for (user, session_id), group in df.groupby(
        ["user", "session_id"],
        sort=False
    ):
        sessions.append(
            (
                str(user),
                str(session_id),
                group["event_type"].to_numpy(dtype=str),
                group["device_type"].to_numpy(dtype=str),
                group["timestamp"].to_numpy(dtype=float),
                group["x"].to_numpy(dtype=float),
                group["y"].to_numpy(dtype=float),
                group["delta_x"].to_numpy(dtype=float),
                group["delta_y"].to_numpy(dtype=float),
            )
        )

    fieldnames = None
    total_rows = 0
    device_rows = 0

    summary = defaultdict(
        lambda: {
            "janelas": 0,
            "mouse": 0,
            "trackpad": 0,
            "mixed": 0,
            "high_confidence": 0,
        }
    )

    # Abre os arquivos uma única vez e escreve por streaming.
    with DATASET_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as dataset_file, DEVICE_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as device_file:

        dataset_writer = None
        device_writer = None

        for ws in WINDOW_SIZES:
            for requested_overlap in REQUESTED_OVERLAPS:

                step = max(
                    1,
                    half_up(ws * (1.0 - requested_overlap))
                )

                effective_overlap = 1.0 - (step / ws)

                config_key = (
                    ws,
                    int(requested_overlap * 100),
                )

                config_count = 0

                for (
                    user,
                    session_id,
                    event_types,
                    device_types,
                    timestamps_ms,
                    x,
                    y,
                    delta_x,
                    delta_y,
                ) in sessions:

                    n = len(event_types)

                    if n < ws:
                        continue

                    window_number = 0

                    for start in range(0, n - ws + 1, step):
                        end = start + ws
                        window_number += 1

                        row = extract_window_features(
                            user=user,
                            session_id=session_id,
                            window_number=window_number,
                            window_size=ws,
                            requested_overlap=requested_overlap,
                            step_events=step,
                            effective_overlap=effective_overlap,
                            event_types=event_types[start:end],
                            device_types=device_types[start:end],
                            timestamps_ms=timestamps_ms[start:end],
                            x=x[start:end],
                            y=y[start:end],
                            delta_x=delta_x[start:end],
                            delta_y=delta_y[start:end],
                        )

                        if dataset_writer is None:
                            fieldnames = list(row.keys())

                            dataset_writer = csv.DictWriter(
                                dataset_file,
                                fieldnames=fieldnames,
                            )
                            dataset_writer.writeheader()

                            device_writer = csv.DictWriter(
                                device_file,
                                fieldnames=fieldnames,
                            )
                            device_writer.writeheader()

                        dataset_writer.writerow(row)

                        total_rows += 1
                        config_count += 1

                        summary[
                            (ws, int(requested_overlap * 100), user)
                        ]["janelas"] += 1

                        context = row["device_context"]

                        if context in ("mouse", "trackpad", "mixed"):
                            summary[
                                (ws, int(requested_overlap * 100), user)
                            ][context] += 1

                        if row["device_high_confidence"] == 1:
                            summary[
                                (ws, int(requested_overlap * 100), user)
                            ]["high_confidence"] += 1

                            # Dataset separado para a ETAPA 1.
                            # Importante: o alvo continua sendo derivado
                            # da heurística do front-end.
                            device_writer.writerow(row)
                            device_rows += 1

                        if total_rows % 25000 == 0:
                            print(
                                f"     {total_rows:,} janelas geradas..."
                                .replace(",", ".")
                            )

                print(
                    f"   ✅ {ws} eventos / "
                    f"{int(requested_overlap*100)}%: "
                    f"{config_count:,} janelas"
                    .replace(",", ".")
                )

    if total_rows == 0:
        raise RuntimeError(
            "Nenhuma janela foi gerada. "
            "Verifique o snapshot e as sessões."
        )

    # ========================================================
    # RESUMO
    # ========================================================

    summary_rows = []

    for (
        window_size,
        overlap_pct,
        user,
    ), values in sorted(summary.items()):

        summary_rows.append(
            {
                "window_size": window_size,
                "requested_overlap_pct": overlap_pct,
                "user": user,
                **values,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # HASH DOS ARQUIVOS GERADOS
    # ========================================================

    dataset_hash = sha256_file(DATASET_PATH)
    device_hash = sha256_file(DEVICE_PATH)

    # Lista automática de colunas de features
    metadata_columns = [
        "target_user",
        "user",
        "session_id",
        "window_id",
        "window_size",
        "requested_overlap_pct",
        "step_events",
        "effective_overlap_pct",
        "window_start_ms",
        "window_end_ms",
        "duration_s",
        "n_events",
        "device_mouse_pct",
        "device_trackpad_pct",
        "device_context",
        "device_confidence",
        "device_high_confidence",
        "device_label_source",
        "target_user_device",
    ]

    feature_columns = [
        c for c in fieldnames
        if c not in metadata_columns
    ]

    manifest = {
        "projeto": "CYBELL-J",
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "fonte": str(RAW_CSV.name),
        "fonte_sha256": sha256_file(RAW_CSV),
        "session_gap_seconds": SESSION_GAP_SECONDS,
        "window_sizes": WINDOW_SIZES,
        "requested_overlaps_pct": [
            int(x * 100) for x in REQUESTED_OVERLAPS
        ],
        "device_context_threshold": DEVICE_CONTEXT_THRESHOLD,
        "device_high_confidence_threshold": DEVICE_HIGH_CONFIDENCE_THRESHOLD,
        "device_label_source": DEVICE_LABEL_SOURCE,
        "total_windows_dataset_mouse": total_rows,
        "total_windows_dataset_device_high_confidence": device_rows,
        "dataset_mouse_sha256": dataset_hash,
        "dataset_device_sha256": device_hash,
        "metadata_columns": metadata_columns,
        "feature_columns": feature_columns,
        "observacoes_metodologicas": [
            "As janelas são formadas dentro de cada sessão e nunca atravessam sessões.",
            "Sessões são separadas após mais de 300 segundos de inatividade.",
            "A cinemática é calculada exclusivamente sobre eventos mouse_move.",
            "Cliques, pressionamentos e scroll possuem métricas próprias.",
            "device_context é contexto/metadado e não deve ser usado como feature no baseline de identidade.",
            "O rótulo mouse/trackpad deriva da heurística do front-end; não representa ground truth físico independente.",
            "dataset_device_high_confidence.csv contém somente janelas com >=95% de consistência do contexto heurístico.",
            "Para avaliar subperfis por dispositivo, recomenda-se comparar o baseline por usuário com uma estratégia condicionada ao device_context.",
        ],
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=4,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 78)
    print("DATASET GERADO COM SUCESSO")
    print("=" * 78)

    print(
        f"\nDataset principal:\n{DATASET_PATH}"
    )
    print(
        f"\nJanelas totais: {total_rows:,}"
        .replace(",", ".")
    )
    print(
        f"SHA-256:\n{dataset_hash}"
    )

    print(
        f"\nDataset de dispositivo (alta confiança):\n{DEVICE_PATH}"
    )
    print(
        f"\nJanelas de alta confiança: {device_rows:,}"
        .replace(",", ".")
    )
    print(
        f"SHA-256:\n{device_hash}"
    )

    print(
        f"\nResumo:\n{SUMMARY_PATH}"
    )

    print(
        f"\nManifesto:\n{MANIFEST_PATH}"
    )

    print(
        "\n✅ MongoDB não foi acessado."
    )
    print(
        "✅ O snapshot RAW não foi modificado."
    )
    print(
        "✅ Dataset preparado para os próximos ensaios "
        "de Random Forest/XGBoost."
    )


if __name__ == "__main__":
    main()