import numpy as np
import pandas as pd


def _percentile(series, q):
    if len(series) == 0:
        return 0.0
    return float(np.percentile(series, q))


def _stats(series):
    series = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()

    if len(series) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "min": 0.0,
        }

    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "median": float(series.median()),
        "p90": _percentile(series, 90),
        "max": float(series.max()),
        "min": float(series.min()),
    }


def extract_mouse_features(events):
    """
    Recebe os eventos de uma janela do mouse e retorna
    um DataFrame com as 89 features esperadas pelo modelo global.
    """

    df = pd.DataFrame(events).copy()

    if df.empty:
        raise ValueError("Não há eventos suficientes para extrair features.")

    # ---------------------------------------------------------
    # Normalização básica
    # ---------------------------------------------------------

    if "timestamp" not in df.columns:
        raise ValueError("Os eventos precisam possuir a coluna 'timestamp'.")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("São necessários pelo menos 2 eventos.")

    # Alguns timestamps podem estar em milissegundos.
    if df["timestamp"].median() > 1e12:
        df["timestamp"] = df["timestamp"] / 1000.0

    # ---------------------------------------------------------
    # Tempo entre eventos
    # ---------------------------------------------------------

    df["dt"] = df["timestamp"].diff() 

    df["dt"] = df["dt"].replace([np.inf, -np.inf], np.nan)

    df.loc[df["dt"] <= 0, "dt"] = np.nan

    # ---------------------------------------------------------
    # Movimento
    # ---------------------------------------------------------

    df["x"] = pd.to_numeric(df.get("x"), errors="coerce")
    df["y"] = pd.to_numeric(df.get("y"), errors="coerce")

    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()

    df["distance"] = np.sqrt(
        df["dx"].fillna(0) ** 2 +
        df["dy"].fillna(0) ** 2
    )

    valid_moves = (
        df["x"].notna()
        & df["y"].notna()
        & df["dx"].notna()
        & df["dy"].notna()
        & df["dt"].notna()
        & (df["dt"] > 0)
    )

    moves = df.loc[valid_moves].copy()

    if not moves.empty:
        moves["speed"] = moves["distance"] / moves["dt"]

        moves["speed_prev"] = moves["speed"].shift(1)

        moves["acceleration"] = (
            moves["speed"] - moves["speed_prev"]
        ) / moves["dt"]

        moves["angle_rad"] = np.arctan2(
            moves["dy"],
            moves["dx"]
        )

        moves["angle_deg"] = (
            np.degrees(moves["angle_rad"]) % 360
        )

        moves["abs_dx"] = moves["dx"].abs()
        moves["abs_dy"] = moves["dy"].abs()

    # ---------------------------------------------------------
    # Duração da janela
    # ---------------------------------------------------------

    duration_s = (
        df["timestamp"].iloc[-1] -
        df["timestamp"].iloc[0]
    )

    duration_s = max(float(duration_s), 1e-9)

    # ---------------------------------------------------------
    # Tipos de evento
    # ---------------------------------------------------------

    event_type = (
        df["event_type"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    n_mouse_move = int((event_type == "mouse_move").sum())
    n_mouse_click = int((event_type == "mouse_click").sum())
    n_mouse_down = int((event_type == "mouse_down").sum())
    n_mouse_up = int((event_type == "mouse_up").sum())
    n_mouse_wheel = int((event_type == "mouse_wheel").sum())

    total_events = len(df)

    event_rate_hz = total_events / duration_s
    move_rate_hz = n_mouse_move / duration_s
    click_rate_hz = n_mouse_click / duration_s
    wheel_rate_hz = n_mouse_wheel / duration_s

    # ---------------------------------------------------------
    # Movimento
    # ---------------------------------------------------------

    movement_steps_valid = len(moves)

    if not moves.empty:
        distance_total_px = float(moves["distance"].sum())

        displacement_px = float(
            np.sqrt(
                (
                    moves["x"].iloc[-1] -
                    moves["x"].iloc[0]
                ) ** 2
                +
                (
                    moves["y"].iloc[-1] -
                    moves["y"].iloc[0]
                ) ** 2
            )
        )

        straightness = (
            displacement_px / distance_total_px
            if distance_total_px > 0
            else 0.0
        )
    else:
        distance_total_px = 0.0
        displacement_px = 0.0
        straightness = 0.0

    # ---------------------------------------------------------
    # Wheel
    # ---------------------------------------------------------

    wheel = df[event_type == "mouse_wheel"].copy()

    if not wheel.empty:
        wheel_dx = pd.to_numeric(
            wheel.get("delta_x"),
            errors="coerce"
        ).fillna(0)

        wheel_dy = pd.to_numeric(
            wheel.get("delta_y"),
            errors="coerce"
        ).fillna(0)

        wheel_horizontal_fraction = float(
            (
                wheel_dx.abs() >
                wheel_dy.abs()
            ).mean()
        )

        wheel_dy_values = wheel_dy.to_numpy()

        if len(wheel_dy_values) > 1:
            signs = np.sign(wheel_dy_values)
            wheel_direction_changes = int(
                np.sum(signs[1:] != signs[:-1])
            )
        else:
            wheel_direction_changes = 0

    else:
        wheel_dx = pd.Series(dtype=float)
        wheel_dy = pd.Series(dtype=float)
        wheel_horizontal_fraction = 0.0
        wheel_direction_changes = 0

    # ---------------------------------------------------------
    # Hold
    # ---------------------------------------------------------

    if "hold_time" in df.columns:
        hold = pd.to_numeric(
            df["hold_time"],
            errors="coerce"
        ).dropna()
    else:
        hold = pd.Series(dtype=float)

    hold = hold[hold >= 0]

    hold_count = len(hold)

    # ---------------------------------------------------------
    # Estatísticas de tempo
    # ---------------------------------------------------------

    event_dt = df["dt"].dropna()

    move_dt = moves["dt"].dropna() if not moves.empty else pd.Series(dtype=float)

    event_stats = _stats(event_dt)
    move_dt_stats = _stats(move_dt)

    # ---------------------------------------------------------
    # Distância
    # ---------------------------------------------------------

    distance_stats = _stats(
        moves["distance"] if not moves.empty else []
    )

    # ---------------------------------------------------------
    # Velocidade
    # ---------------------------------------------------------

    speed_stats = _stats(
        moves["speed"] if not moves.empty else []
    )

    # ---------------------------------------------------------
    # Aceleração
    # ---------------------------------------------------------

    accel = (
        moves["acceleration"]
        if not moves.empty
        else pd.Series(dtype=float)
    )

    accel_stats = _stats(accel)
    abs_accel_stats = _stats(accel.abs())

    # ---------------------------------------------------------
    # Ângulo
    # ---------------------------------------------------------

    if not moves.empty:
        angles = moves["angle_deg"].dropna()

        if len(angles) > 1:
            angle_diff = np.diff(angles.to_numpy())
            angle_diff = (angle_diff + 180) % 360 - 180
            turn_abs = np.abs(angle_diff)
        else:
            turn_abs = []

    else:
        turn_abs = []

    turn_stats = _stats(turn_abs)

    # ---------------------------------------------------------
    # DX / DY
    # ---------------------------------------------------------

    dx_stats = _stats(
        moves["abs_dx"]
        if not moves.empty
        else []
    )

    dy_stats = _stats(
        moves["abs_dy"]
        if not moves.empty
        else []
    )

    # ---------------------------------------------------------
    # Click intervals
    # ---------------------------------------------------------

    clicks = df.loc[event_type == "mouse_click", "timestamp"]

    if len(clicks) > 1:
        click_intervals = clicks.diff().dropna()
    else:
        click_intervals = pd.Series(dtype=float)

    click_stats = _stats(click_intervals)

    # ---------------------------------------------------------
    # Hold time
    # ---------------------------------------------------------

    hold_stats = _stats(hold)

    # ---------------------------------------------------------
    # Wheel statistics
    # ---------------------------------------------------------

    wheel_abs_dx_stats = _stats(wheel_dx.abs())
    wheel_abs_dy_stats = _stats(wheel_dy.abs())

    wheel_dy_stats = _stats(wheel_dy)

    # ---------------------------------------------------------
    # Montagem das 89 features
    # ---------------------------------------------------------

    features = {
        "duration_s": duration_s,

        "n_mouse_move": n_mouse_move,
        "n_mouse_click": n_mouse_click,
        "n_mouse_down": n_mouse_down,
        "n_mouse_up": n_mouse_up,
        "n_mouse_wheel": n_mouse_wheel,

        "event_rate_hz": event_rate_hz,
        "move_rate_hz": move_rate_hz,
        "click_rate_hz": click_rate_hz,
        "wheel_rate_hz": wheel_rate_hz,

        "movement_steps_valid": movement_steps_valid,
        "distance_total_px": distance_total_px,
        "displacement_px": displacement_px,
        "straightness": straightness,

        "wheel_horizontal_fraction": wheel_horizontal_fraction,
        "wheel_direction_changes": wheel_direction_changes,

        "hold_count": hold_count,

        "event_dt_s_mean": event_stats["mean"],
        "event_dt_s_std": event_stats["std"],
        "event_dt_s_median": event_stats["median"],
        "event_dt_s_p90": event_stats["p90"],
        "event_dt_s_max": event_stats["max"],

        "move_dt_s_mean": move_dt_stats["mean"],
        "move_dt_s_std": move_dt_stats["std"],
        "move_dt_s_median": move_dt_stats["median"],
        "move_dt_s_p90": move_dt_stats["p90"],
        "move_dt_s_max": move_dt_stats["max"],

        "distance_px_mean": distance_stats["mean"],
        "distance_px_std": distance_stats["std"],
        "distance_px_median": distance_stats["median"],
        "distance_px_p90": distance_stats["p90"],
        "distance_px_max": distance_stats["max"],

        "speed_px_s_mean": speed_stats["mean"],
        "speed_px_s_std": speed_stats["std"],
        "speed_px_s_median": speed_stats["median"],
        "speed_px_s_p90": speed_stats["p90"],
        "speed_px_s_max": speed_stats["max"],

        "accel_px_s2_mean": accel_stats["mean"],
        "accel_px_s2_std": accel_stats["std"],
        "accel_px_s2_median": accel_stats["median"],
        "accel_px_s2_p90": accel_stats["p90"],
        "accel_px_s2_max": accel_stats["max"],
        "accel_px_s2_min": accel_stats["min"],

        "abs_accel_px_s2_mean": abs_accel_stats["mean"],
        "abs_accel_px_s2_std": abs_accel_stats["std"],
        "abs_accel_px_s2_median": abs_accel_stats["median"],
        "abs_accel_px_s2_p90": abs_accel_stats["p90"],
        "abs_accel_px_s2_max": abs_accel_stats["max"],

        "turn_abs_deg_mean": turn_stats["mean"],
        "turn_abs_deg_std": turn_stats["std"],
        "turn_abs_deg_median": turn_stats["median"],
        "turn_abs_deg_p90": turn_stats["p90"],
        "turn_abs_deg_max": turn_stats["max"],

        "move_abs_dx_px_mean": dx_stats["mean"],
        "move_abs_dx_px_std": dx_stats["std"],
        "move_abs_dx_px_median": dx_stats["median"],
        "move_abs_dx_px_p90": dx_stats["p90"],
        "move_abs_dx_px_max": dx_stats["max"],

        "move_abs_dy_px_mean": dy_stats["mean"],
        "move_abs_dy_px_std": dy_stats["std"],
        "move_abs_dy_px_median": dy_stats["median"],
        "move_abs_dy_px_p90": dy_stats["p90"],
        "move_abs_dy_px_max": dy_stats["max"],

        "click_interval_s_mean": click_stats["mean"],
        "click_interval_s_std": click_stats["std"],
        "click_interval_s_median": click_stats["median"],
        "click_interval_s_p90": click_stats["p90"],
        "click_interval_s_max": click_stats["max"],

        "hold_ms_mean": hold_stats["mean"],
        "hold_ms_std": hold_stats["std"],
        "hold_ms_median": hold_stats["median"],
        "hold_ms_p90": hold_stats["p90"],
        "hold_ms_max": hold_stats["max"],

        "wheel_abs_dx_mean": wheel_abs_dx_stats["mean"],
        "wheel_abs_dx_std": wheel_abs_dx_stats["std"],
        "wheel_abs_dx_median": wheel_abs_dx_stats["median"],
        "wheel_abs_dx_p90": wheel_abs_dx_stats["p90"],
        "wheel_abs_dx_max": wheel_abs_dx_stats["max"],

        "wheel_abs_dy_mean": wheel_abs_dy_stats["mean"],
        "wheel_abs_dy_std": wheel_abs_dy_stats["std"],
        "wheel_abs_dy_median": wheel_abs_dy_stats["median"],
        "wheel_abs_dy_p90": wheel_abs_dy_stats["p90"],
        "wheel_abs_dy_max": wheel_abs_dy_stats["max"],

        "wheel_dy_mean": wheel_dy_stats["mean"],
        "wheel_dy_std": wheel_dy_stats["std"],
        "wheel_dy_median": wheel_dy_stats["median"],
        "wheel_dy_p90": wheel_dy_stats["p90"],
        "wheel_dy_max": wheel_dy_stats["max"],
        "wheel_dy_min": wheel_dy_stats["min"],
    }

    return pd.DataFrame([features])