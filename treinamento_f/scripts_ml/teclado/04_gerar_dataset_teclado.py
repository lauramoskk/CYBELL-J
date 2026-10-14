"""
04_gerar_dataset_teclado.py

Gera o dataset de características comportamentais do teclado do CYBELL-J.

Mantém o mesmo desenho experimental usado no mouse:
- sessões já definidas por pausa > 300 s;
- janelas de 25, 30, 50 e 100 eventos;
- overlaps solicitados de 25%, 50% e 75%;
- janelas nunca atravessam session_id;
- metadados de janela separados das features comportamentais.

O teclado possui features próprias. O protocolo experimental, porém,
permanece equivalente ao do mouse.

Entrada:
data/teclado_dataset/keystrokes_com_sessoes.csv

Saídas:
data/teclado_dataset/
    dataset_teclado.csv
    resumo_dataset_teclado.csv
    dataset_teclado_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = BASE_DIR / "data" / "teclado_dataset"
SNAPSHOT_DIR = BASE_DIR / "data" / "teclado_snapshot"

INPUT_CSV = DATASET_DIR / "keystrokes_com_sessoes.csv"
SESSION_META = DATASET_DIR / "sessoes_teclado_metadata.json"
SNAPSHOT_META = SNAPSHOT_DIR / "keystrokes_snapshot_metadata.json"

OUT_DATASET = DATASET_DIR / "dataset_teclado.csv"
OUT_SUMMARY = DATASET_DIR / "resumo_dataset_teclado.csv"
OUT_MANIFEST = DATASET_DIR / "dataset_teclado_manifest.json"

WINDOW_SIZES = [25, 30, 50, 100]
REQUESTED_OVERLAPS = [0.25, 0.50, 0.75]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def safe_rate(count: int, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return float(count / duration_s)


def stats_basic(
    values,
    prefix: str,
    include_min: bool = True,
) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    result = {
        f"{prefix}_mean": 0.0,
        f"{prefix}_std": 0.0,
        f"{prefix}_median": 0.0,
        f"{prefix}_p25": 0.0,
        f"{prefix}_p75": 0.0,
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
    result[f"{prefix}_p25"] = float(np.percentile(arr, 25))
    result[f"{prefix}_p75"] = float(np.percentile(arr, 75))
    result[f"{prefix}_p90"] = float(np.percentile(arr, 90))
    result[f"{prefix}_max"] = float(np.max(arr))

    if include_min:
        result[f"{prefix}_min"] = float(np.min(arr))

    return result


def normalize_key(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            "" if pd.isna(v) else str(v).strip().lower()
            for v in values
        ],
        dtype=object,
    )


def startswith_any(value: str, prefixes: tuple[str, ...]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def extract_window_features(
    user: str,
    session_id: str,
    window_number: int,
    window_size: int,
    requested_overlap: float,
    step_events: int,
    effective_overlap: float,
    timestamps_ms: np.ndarray,
    event_types: np.ndarray,
    keys: np.ndarray,
    hold_times: np.ndarray,
) -> dict:
    n_events = len(event_types)

    start_ms = float(timestamps_ms[0])
    end_ms = float(timestamps_ms[-1])
    duration_s = max(0.0, (end_ms - start_ms) / 1000.0)

    event_types = np.char.lower(
        np.asarray(event_types, dtype=str)
    )
    keys_norm = normalize_key(keys)

    keydown_mask = event_types == "keydown"
    shortcut_mask = event_types == "shortcut"

    n_keydown = int(np.sum(keydown_mask))
    n_shortcut = int(np.sum(shortcut_mask))

    # Intervalos entre todos os eventos da janela.
    event_dt_s = np.diff(timestamps_ms) / 1000.0
    event_dt_s = event_dt_s[
        np.isfinite(event_dt_s) & (event_dt_s > 0)
    ]

    # Intervalos somente entre keydowns.
    keydown_t = timestamps_ms[keydown_mask]
    keydown_dt_s = np.diff(keydown_t) / 1000.0
    keydown_dt_s = keydown_dt_s[
        np.isfinite(keydown_dt_s) & (keydown_dt_s > 0)
    ]

    keydown_keys = keys_norm[keydown_mask]

    # Teclas funcionais/comportamentais.
    backspace_count = int(
        np.sum([k.startswith("back") for k in keydown_keys])
    )
    space_count = int(
        np.sum([k.startswith("space") for k in keydown_keys])
    )
    shift_count = int(
        np.sum([k.startswith("shift") for k in keydown_keys])
    )
    capslock_count = int(
        np.sum([k.startswith("caps") for k in keydown_keys])
    )
    enter_count = int(
        np.sum([k.startswith("enter") for k in keydown_keys])
    )
    tab_count = int(
        np.sum([k.startswith("tab") for k in keydown_keys])
    )
    delete_count = int(
        np.sum([k.startswith("delet") for k in keydown_keys])
    )
    arrow_count = int(
        np.sum([k.startswith("arrow") for k in keydown_keys])
    )
    digit_count = int(
        np.sum([k.startswith("digit") for k in keydown_keys])
    )
    numpad_count = int(
        np.sum(
            [
                startswith_any(k, ("numpa", "numlo"))
                for k in keydown_keys
            ]
        )
    )
    modifier_count = int(
        np.sum(
            [
                startswith_any(
                    k,
                    ("shift", "alt", "meta", "contr", "ctrl"),
                )
                for k in keydown_keys
            ]
        )
    )
    punctuation_count = int(
        np.sum(
            [
                startswith_any(
                    k,
                    (
                        "perio",
                        "comma",
                        "quote",
                        "semic",
                        "slash",
                        "brack",
                        "minus",
                        "equal",
                        "backq",
                    ),
                )
                for k in keydown_keys
            ]
        )
    )

    denom_keys = max(n_keydown, 1)

    # Hold time existe nos eventos shortcut da coleta atual.
    shortcut_hold_ms = pd.to_numeric(
        pd.Series(hold_times[shortcut_mask]),
        errors="coerce",
    ).to_numpy(dtype=float)

    shortcut_hold_ms = shortcut_hold_ms[
        np.isfinite(shortcut_hold_ms)
        & (shortcut_hold_ms >= 0)
    ]

    # Características de ritmo em diferentes escalas.
    if keydown_dt_s.size:
        fast_interval_fraction = float(
            np.mean(keydown_dt_s <= 0.25)
        )
        pause_05s_fraction = float(
            np.mean(keydown_dt_s > 0.50)
        )
        pause_1s_fraction = float(
            np.mean(keydown_dt_s > 1.00)
        )
        pause_2s_fraction = float(
            np.mean(keydown_dt_s > 2.00)
        )
    else:
        fast_interval_fraction = 0.0
        pause_05s_fraction = 0.0
        pause_1s_fraction = 0.0
        pause_2s_fraction = 0.0

    row = {
        # Identificação / alvo.
        "target_user": user,
        "user": user,
        "session_id": session_id,
        "window_id": (
            f"{session_id}_W{window_size}_"
            f"O{int(requested_overlap * 100):02d}_"
            f"{window_number:06d}"
        ),

        # Configuração experimental.
        "window_size": int(window_size),
        "requested_overlap_pct": int(
            round(requested_overlap * 100)
        ),
        "step_events": int(step_events),
        "effective_overlap_pct": round(
            effective_overlap * 100,
            4,
        ),

        # Tempo.
        "window_start_ms": int(start_ms),
        "window_end_ms": int(end_ms),
        "duration_s": float(duration_s),

        # Contagens.
        "n_events": int(n_events),
        "n_keydown": n_keydown,
        "n_shortcut": n_shortcut,

        # Taxas temporais.
        "event_rate_hz": safe_rate(n_events, duration_s),
        "keydown_rate_hz": safe_rate(n_keydown, duration_s),
        "shortcut_rate_hz": safe_rate(n_shortcut, duration_s),

        # Teclas funcionais.
        "backspace_count": backspace_count,
        "backspace_fraction": backspace_count / denom_keys,
        "space_count": space_count,
        "space_fraction": space_count / denom_keys,
        "shift_count": shift_count,
        "shift_fraction": shift_count / denom_keys,
        "capslock_count": capslock_count,
        "capslock_fraction": capslock_count / denom_keys,
        "enter_count": enter_count,
        "enter_fraction": enter_count / denom_keys,
        "tab_count": tab_count,
        "tab_fraction": tab_count / denom_keys,
        "delete_count": delete_count,
        "delete_fraction": delete_count / denom_keys,
        "arrow_count": arrow_count,
        "arrow_fraction": arrow_count / denom_keys,
        "digit_count": digit_count,
        "digit_fraction": digit_count / denom_keys,
        "numpad_count": numpad_count,
        "numpad_fraction": numpad_count / denom_keys,
        "modifier_count": modifier_count,
        "modifier_fraction": modifier_count / denom_keys,
        "punctuation_count": punctuation_count,
        "punctuation_fraction": punctuation_count / denom_keys,

        # Ritmo.
        "fast_interval_fraction_le_025s": fast_interval_fraction,
        "pause_fraction_gt_05s": pause_05s_fraction,
        "pause_fraction_gt_1s": pause_1s_fraction,
        "pause_fraction_gt_2s": pause_2s_fraction,

        # Shortcuts.
        "shortcut_hold_count": int(shortcut_hold_ms.size),
    }

    row.update(
        stats_basic(
            event_dt_s,
            "event_dt_s",
        )
    )
    row.update(
        stats_basic(
            keydown_dt_s,
            "keydown_dt_s",
        )
    )
    row.update(
        stats_basic(
            shortcut_hold_ms,
            "shortcut_hold_ms",
        )
    )

    return row


def load_data() -> tuple[pd.DataFrame, dict]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo com sessões não encontrado:\n{INPUT_CSV}"
        )

    if not SESSION_META.exists():
        raise FileNotFoundError(
            f"Metadados das sessões não encontrados:\n{SESSION_META}"
        )

    with SESSION_META.open("r", encoding="utf-8") as f:
        session_meta = json.load(f)

    df = pd.read_csv(
        INPUT_CSV,
        low_memory=False,
    )

    required = [
        "user",
        "timestamp",
        "event_type",
        "key",
        "hold_time",
        "session_id",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(missing)
        )

    df = df.copy()

    df["user"] = (
        df["user"]
        .astype("string")
        .str.strip()
    )

    df["session_id"] = (
        df["session_id"]
        .astype("string")
        .str.strip()
    )

    df["event_type"] = (
        df["event_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    df["hold_time"] = pd.to_numeric(
        df["hold_time"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "user",
            "session_id",
            "event_type",
            "timestamp",
        ]
    ).copy()

    order_cols = ["user", "session_id", "timestamp"]

    if "_id" in df.columns:
        df["_id"] = df["_id"].astype(str)
        order_cols.append("_id")

    df = df.sort_values(
        order_cols
    ).reset_index(drop=True)

    return df, session_meta


def main() -> None:
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("CYBELL-J — GERAÇÃO DO DATASET DE TECLADO")
    print("=" * 100)

    df, session_meta = load_data()

    print(f"\nEventos carregados: {len(df):,}".replace(",", "."))
    print(f"Usuárias: {df['user'].nunique()}")
    print(f"Sessões: {df['session_id'].nunique()}")

    print("\nConfigurações de janela:")

    configurations = []

    for ws in WINDOW_SIZES:
        for requested_overlap in REQUESTED_OVERLAPS:
            step = max(
                1,
                half_up(
                    ws * (1.0 - requested_overlap)
                ),
            )
            effective_overlap = 1.0 - (step / ws)

            configurations.append(
                (
                    ws,
                    requested_overlap,
                    step,
                    effective_overlap,
                )
            )

            print(
                f"  {ws:>3} eventos | "
                f"overlap solicitado "
                f"{int(requested_overlap * 100):>2}% | "
                f"passo {step:>3} | "
                f"overlap efetivo "
                f"{effective_overlap * 100:>6.2f}%"
            )

    sessions = []

    for (user, session_id), group in df.groupby(
        ["user", "session_id"],
        sort=False,
    ):
        sessions.append(
            (
                str(user),
                str(session_id),
                group["timestamp"].to_numpy(dtype=float),
                group["event_type"].astype(str).to_numpy(),
                group["key"].to_numpy(dtype=object),
                group["hold_time"].to_numpy(dtype=float),
            )
        )

    rows = []
    summary = defaultdict(int)

    print("\nGerando janelas e features...")

    for (
        window_size,
        requested_overlap,
        step_events,
        effective_overlap,
    ) in configurations:

        config_rows = 0

        for (
            user,
            session_id,
            timestamps_ms,
            event_types,
            keys,
            hold_times,
        ) in sessions:

            n = len(timestamps_ms)

            if n < window_size:
                continue

            window_number = 0

            for start in range(
                0,
                n - window_size + 1,
                step_events,
            ):
                end = start + window_size

                row = extract_window_features(
                    user=user,
                    session_id=session_id,
                    window_number=window_number,
                    window_size=window_size,
                    requested_overlap=requested_overlap,
                    step_events=step_events,
                    effective_overlap=effective_overlap,
                    timestamps_ms=timestamps_ms[start:end],
                    event_types=event_types[start:end],
                    keys=keys[start:end],
                    hold_times=hold_times[start:end],
                )

                rows.append(row)
                config_rows += 1
                window_number += 1

        key = (
            int(window_size),
            int(round(requested_overlap * 100)),
        )
        summary[key] = config_rows

        print(
            f"  janela={window_size:>3} | "
            f"overlap={int(requested_overlap * 100):>2}% "
            f"-> {config_rows:,} janelas".replace(",", ".")
        )

    if not rows:
        raise RuntimeError(
            "Nenhuma janela foi gerada."
        )

    dataset = pd.DataFrame(rows)

    # Validações.
    if dataset["window_id"].duplicated().any():
        duplicates = int(
            dataset["window_id"].duplicated().sum()
        )
        raise RuntimeError(
            f"Foram encontrados {duplicates} window_id duplicados."
        )

    if dataset["session_id"].isna().any():
        raise RuntimeError(
            "Há janelas sem session_id."
        )

    # Cada janela deve pertencer integralmente a uma única sessão
    # porque a geração foi feita dentro do groupby de session_id.
    original_sessions = set(
        df["session_id"].astype(str).unique()
    )
    generated_sessions = set(
        dataset["session_id"].astype(str).unique()
    )

    if not generated_sessions.issubset(original_sessions):
        raise RuntimeError(
            "Há session_id no dataset que não existe na base de eventos."
        )

    dataset.to_csv(
        OUT_DATASET,
        index=False,
        encoding="utf-8",
    )

    summary_rows = []

    for (
        window_size,
        requested_overlap_pct,
    ), count in sorted(summary.items()):

        step = max(
            1,
            half_up(
                window_size
                * (
                    1.0
                    - requested_overlap_pct / 100.0
                )
            ),
        )
        effective = (
            1.0 - step / window_size
        ) * 100.0

        sub = dataset[
            (dataset["window_size"] == window_size)
            & (
                dataset["requested_overlap_pct"]
                == requested_overlap_pct
            )
        ]

        summary_rows.append(
            {
                "window_size": window_size,
                "requested_overlap_pct": requested_overlap_pct,
                "step_events": step,
                "effective_overlap_pct": effective,
                "janelas": int(count),
                "usuarios": int(
                    sub["user"].nunique()
                ),
                "sessoes_representadas": int(
                    sub["session_id"].nunique()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    # Colunas consideradas metadados nos experimentos.
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
        "n_events",
    ]

    feature_columns = [
        c
        for c in dataset.columns
        if c not in metadata_columns
        and pd.api.types.is_numeric_dtype(dataset[c])
    ]

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(
            INPUT_CSV.relative_to(BASE_DIR)
        ),
        "source_snapshot_sha256": session_meta.get(
            "source_sha256"
        ),
        "window_sizes": WINDOW_SIZES,
        "requested_overlaps_pct": [
            int(x * 100)
            for x in REQUESTED_OVERLAPS
        ],
        "rows": int(len(dataset)),
        "users": int(dataset["user"].nunique()),
        "sessions_represented": int(
            dataset["session_id"].nunique()
        ),
        "metadata_columns": metadata_columns,
        "feature_columns": feature_columns,
        "n_features_numeric": int(
            len(feature_columns)
        ),
        "important": (
            "As features do teclado são próprias da modalidade. "
            "Os experimentos devem usar o mesmo protocolo de "
            "validação e pré-processamento do mouse."
        ),
        "next_step": (
            "Executar o Experimento 2 global com RF e XGBoost "
            "usando 3 folds por session_id."
        ),
    }

    with OUT_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 100)
    print("DATASET DE TECLADO CONCLUÍDO")
    print("=" * 100)

    print(f"\nJanelas totais: {len(dataset):,}".replace(",", "."))
    print(
        f"Features numéricas comportamentais: "
        f"{len(feature_columns)}"
    )

    print("\nArquivos gerados:")
    print(f"  {OUT_DATASET}")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_MANIFEST}")

    print("\nValidações:")
    print("  Nenhuma janela atravessa session_id.")
    print("  window_id é único.")
    print("  As 12 configurações de janela/overlap foram geradas.")

    print(
        "\nAinda não foi treinado nenhum modelo."
    )


if __name__ == "__main__":
    main()
