import numpy as np
import pandas as pd


WINDOW_SIZE = 50


def safe_rate(count: int, duration_s: float) -> float:
    """Calcula a frequência de eventos por segundo."""
    if duration_s <= 0:
        return 0.0

    return float(count / duration_s)


def stats_basic(
    values,
    prefix: str,
    include_min: bool = True,
) -> dict:
    """Calcula as estatísticas usadas no treinamento."""

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


def normalize_key(value) -> str:
    """Normaliza o código da tecla como no treinamento."""

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def startswith_any(
    value: str,
    prefixes: tuple[str, ...],
) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def extract_keyboard_features(eventos) -> pd.DataFrame:
    """
    Transforma eventos brutos do teclado em uma linha
    com as características esperadas pelo modelo treinado.

    Os timestamps e hold_time são interpretados em milissegundos,
    como na coleta atual do frontend.
    """

    if not isinstance(eventos, (list, tuple)):
        raise ValueError("Os eventos devem ser fornecidos em uma lista.")

    eventos_validos = []

    for evento in eventos:
        if not isinstance(evento, dict):
            continue

        try:
            timestamp = float(evento["timestamp"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

        if not np.isfinite(timestamp):
            continue

        event_type = str(evento.get("event_type", "")).strip().lower()

        if not event_type:
            continue

        eventos_validos.append({
            "timestamp": timestamp,
            "event_type": event_type,
            "key": normalize_key(evento.get("key", "")),
            "hold_time": evento.get("hold_time", np.nan),
        })

    # Mesma ordem temporal usada na preparação do dataset.
    eventos_validos.sort(key=lambda evento: evento["timestamp"])

    if len(eventos_validos) < WINDOW_SIZE:
        raise ValueError(
            f"São necessários pelo menos {WINDOW_SIZE} eventos válidos "
            f"de teclado; foram recebidos {len(eventos_validos)}."
        )

    # Utiliza a janela mais recente de 50 eventos.
    eventos_validos = eventos_validos[-WINDOW_SIZE:]

    timestamps_ms = np.asarray(
        [evento["timestamp"] for evento in eventos_validos],
        dtype=float,
    )

    event_types = np.asarray(
        [evento["event_type"] for evento in eventos_validos],
        dtype=str,
    )

    keys_norm = np.asarray(
        [evento["key"] for evento in eventos_validos],
        dtype=object,
    )

    hold_times = np.asarray(
        [evento["hold_time"] for evento in eventos_validos],
        dtype=object,
    )

    n_events = len(event_types)

    start_ms = float(timestamps_ms[0])
    end_ms = float(timestamps_ms[-1])

    duration_s = max(
        0.0,
        (end_ms - start_ms) / 1000.0,
    )

    keydown_mask = event_types == "keydown"
    shortcut_mask = event_types == "shortcut"

    n_keydown = int(np.sum(keydown_mask))
    n_shortcut = int(np.sum(shortcut_mask))

    # Intervalos entre todos os eventos.
    event_dt_s = np.diff(timestamps_ms) / 1000.0
    event_dt_s = event_dt_s[
        np.isfinite(event_dt_s) & (event_dt_s > 0)
    ]

    # Intervalos apenas entre keydowns.
    keydown_t = timestamps_ms[keydown_mask]
    keydown_dt_s = np.diff(keydown_t) / 1000.0
    keydown_dt_s = keydown_dt_s[
        np.isfinite(keydown_dt_s) & (keydown_dt_s > 0)
    ]

    keydown_keys = keys_norm[keydown_mask]

    # Contagem das teclas comportamentais.
    backspace_count = int(sum(
        key.startswith("back") for key in keydown_keys
    ))

    space_count = int(sum(
        key.startswith("space") for key in keydown_keys
    ))

    shift_count = int(sum(
        key.startswith("shift") for key in keydown_keys
    ))

    capslock_count = int(sum(
        key.startswith("caps") for key in keydown_keys
    ))

    enter_count = int(sum(
        key.startswith("enter") for key in keydown_keys
    ))

    tab_count = int(sum(
        key.startswith("tab") for key in keydown_keys
    ))

    delete_count = int(sum(
        key.startswith("delet") for key in keydown_keys
    ))

    arrow_count = int(sum(
        key.startswith("arrow") for key in keydown_keys
    ))

    digit_count = int(sum(
        key.startswith("digit") for key in keydown_keys
    ))

    numpad_count = int(sum(
        startswith_any(key, ("numpa", "numlo"))
        for key in keydown_keys
    ))

    modifier_count = int(sum(
        startswith_any(
            key,
            ("shift", "alt", "meta", "contr", "ctrl"),
        )
        for key in keydown_keys
    ))

    punctuation_count = int(sum(
        startswith_any(
            key,
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
        for key in keydown_keys
    ))

    denom_keys = max(n_keydown, 1)

    # Tempo de retenção dos atalhos, em milissegundos.
    shortcut_hold_ms = pd.to_numeric(
        pd.Series(hold_times[shortcut_mask]),
        errors="coerce",
    ).to_numpy(dtype=float)

    shortcut_hold_ms = shortcut_hold_ms[
        np.isfinite(shortcut_hold_ms)
        & (shortcut_hold_ms >= 0)
    ]

    # Características de ritmo.
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

    # Calcula as mesmas características do treinamento.
    row = {
        "duration_s": float(duration_s),
        "n_keydown": n_keydown,
        "n_shortcut": n_shortcut,

        "event_rate_hz": safe_rate(n_events, duration_s),
        "keydown_rate_hz": safe_rate(n_keydown, duration_s),
        "shortcut_rate_hz": safe_rate(n_shortcut, duration_s),

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

        "fast_interval_fraction_le_025s": fast_interval_fraction,
        "pause_fraction_gt_05s": pause_05s_fraction,
        "pause_fraction_gt_1s": pause_1s_fraction,
        "pause_fraction_gt_2s": pause_2s_fraction,

        "shortcut_hold_count": int(shortcut_hold_ms.size),
    }

    row.update(stats_basic(event_dt_s, "event_dt_s"))
    row.update(stats_basic(keydown_dt_s, "keydown_dt_s"))
    row.update(stats_basic(shortcut_hold_ms, "shortcut_hold_ms"))

    return pd.DataFrame([row])