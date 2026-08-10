"""02_calcular_metricas.py

Conecta ao MongoDB (lendo MONGO_URI do .env), carrega as coleções `mouse_events`
e `keystrokes`, calcula métricas de velocidade, aceleração e trajetória para eventos
de mouse, e recalcula `hold_time` e `flight_time` para keystrokes quando necessário.

Salva os DataFrames com métricas em CSV na pasta `data/`.

Uso: python 02_calcular_metricas.py
"""
from pymongo import MongoClient
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

load_dotenv()

URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB", "cybell_db")

if not URI:
    raise RuntimeError("MONGO_URI não está definida. Defina em .env ou como variável de ambiente.")


def get_db(uri: str, db_name: str):
    client = MongoClient(uri)
    return client[db_name]


def fetch_collection_as_df(db, coll_name: str) -> pd.DataFrame:
    docs = list(db[coll_name].find())
    if not docs:
        return pd.DataFrame()
    # Convert ObjectId to string
    for d in docs:
        if "_id" in d:
            d["_id"] = str(d["_id"])
    df = pd.json_normalize(docs)
    return df


def detect_columns(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_seconds(ts: pd.Series) -> pd.Series:
    # If timestamps look like milliseconds (>=1e12), convert to seconds
    if ts.max(skipna=True) > 1e12:
        return ts.astype(float) / 1000.0
    return ts.astype(float)


def normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "user_id" not in df.columns:
        df["user_id"] = pd.NA
    if "user" in df.columns:
        df["user_id"] = df["user_id"].fillna(df["user"])
    if "data.user" in df.columns:
        df["user_id"] = df["user_id"].fillna(df["data.user"])

    if "session_id" not in df.columns:
        df["session_id"] = pd.NA
    if "data.session_id" in df.columns:
        df["session_id"] = df["session_id"].fillna(df["data.session_id"])

    for col in ["user_id", "session_id"]:
        if col in df.columns:
            s = df[col].astype("string").str.strip()
            s = s.replace({"": pd.NA, "nan": pd.NA})
            df[col] = s

    return df


def infer_session_ids(df: pd.DataFrame, time_cols: list[str], threshold_seconds: float, prefix: str) -> pd.DataFrame:
    df = df.copy()
    if "session_id" not in df.columns:
        df["session_id"] = pd.NA

    time_col = detect_columns(df, time_cols)
    if time_col is None:
        return df

    df["_t"] = ensure_seconds(pd.to_numeric(df[time_col], errors="coerce"))
    df["user_key"] = df.get("user_id", pd.Series([pd.NA] * len(df))).astype("string").fillna("__unknown")
    df = df.sort_values(["user_key", "_t"], na_position="last").copy()

    records = []
    for user, sub in df.groupby("user_key", dropna=False):
        sub = sub.copy()
        sub["_t"] = pd.to_numeric(sub["_t"], errors="coerce")
        sub["time_gap"] = sub["_t"].diff()
        sub["new_block"] = sub["time_gap"].gt(threshold_seconds) | sub["_t"].isna()
        sub["session_block"] = sub["new_block"].cumsum().astype(int)

        sub["session_id"] = sub["session_id"].astype("string")
        sub["missing_session"] = sub["session_id"].isna() | (sub["session_id"] == "")
        sub["prev_valid_session"] = sub["session_id"].where(~sub["missing_session"]).ffill()
        same_as_prev = sub["missing_session"] & sub["time_gap"].le(threshold_seconds) & sub["prev_valid_session"].notna()
        sub.loc[same_as_prev, "session_id"] = sub.loc[same_as_prev, "prev_valid_session"]

        remaining_missing = sub["session_id"].isna() | (sub["session_id"] == "")
        sub.loc[remaining_missing, "session_id"] = [
            f"{user}_{prefix}_{block}"
            for block in sub.loc[remaining_missing, "session_block"]
        ]

        records.append(sub)

    return pd.concat(records, ignore_index=True)


def process_mouse_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    t_col = detect_columns(df, ["data.timestamp", "timestamp", "data.time", "time"])
    x_col = detect_columns(df, ["data.x", "x", "data.position.x"])
    y_col = detect_columns(df, ["data.y", "y", "data.position.y"])

    if not t_col or not x_col or not y_col:
        raise RuntimeError("Não foi possível localizar colunas de timestamp/x/y em mouse_events")

    df = normalize_identifiers(df)
    df = infer_session_ids(df, ["data.timestamp", "timestamp", "data.time", "time"], threshold_seconds=60.0, prefix="mouse")
    df = df.copy()
    df["_t"] = ensure_seconds(pd.to_numeric(df[t_col], errors="coerce"))
    df["_x"] = pd.to_numeric(df[x_col], errors="coerce").astype(float)
    df["_y"] = pd.to_numeric(df[y_col], errors="coerce").astype(float)

    # Sort by user/session/timestamp if available
    group_cols = [c for c in ("user_id", "session_id") if c in df.columns]
    df.sort_values(group_cols + ["_t"], inplace=True)

    # Compute deltas per group
    df["dt"] = df.groupby(group_cols)["_t"].diff()
    df["dx"] = df.groupby(group_cols)["_x"].diff()
    df["dy"] = df.groupby(group_cols)["_y"].diff()
    df["distance"] = np.sqrt(df["dx"].fillna(0) ** 2 + df["dy"].fillna(0) ** 2)

    # speed (pixels / second)
    df["speed"] = df["distance"] / df["dt"]
    df.loc[~np.isfinite(df["speed"]), "speed"] = np.nan

    # acceleration (change in speed / dt)
    df["speed_prev"] = df.groupby(group_cols)["speed"].shift(1)
    df["acceleration"] = (df["speed"] - df["speed_prev"]) / df["dt"]
    df.loc[~np.isfinite(df["acceleration"]), "acceleration"] = np.nan

    # trajectory angle (degrees)
    df["angle_rad"] = np.arctan2(df["dy"], df["dx"])
    df["angle_deg"] = np.degrees(df["angle_rad"]).mod(360)

    # cumulative distance per group
    df["cum_distance"] = df.groupby(group_cols)["distance"].cumsum()

    # Drop helper columns we don't want in final CSV (keep dt,distance,speed,acceleration,angle_deg)
    return df


def process_keystrokes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    press_col = detect_columns(df, ["data.press_time", "data.pressTime", "press_time", "pressTime", "press"])
    release_col = detect_columns(df, ["data.release_time", "data.releaseTime", "release_time", "releaseTime", "release"])

    if not press_col or not release_col:
        print("Aviso: não foi possível localizar press/release em keystrokes; retornando original")
        return df

    df = normalize_identifiers(df)
    df = infer_session_ids(df, ["data.press_time", "press_time", "pressTime", "press"], threshold_seconds=60.0, prefix="key")
    df = df.copy()
    df["press_t"] = ensure_seconds(pd.to_numeric(df[press_col], errors="coerce"))
    df["release_t"] = ensure_seconds(pd.to_numeric(df[release_col], errors="coerce"))

    group_cols = [c for c in ("user_id", "session_id") if c in df.columns]
    df.sort_values(group_cols + ["press_t"], inplace=True)

    # Hold time in milliseconds
    df["hold_time_ms_calc"] = (df["release_t"] - df["press_t"]) * 1000.0

    # Flight time: press_current - release_prev (ms)
    df["release_prev"] = df.groupby(group_cols)["release_t"].shift(1)
    df["flight_time_ms_calc"] = (df["press_t"] - df["release_prev"]) * 1000.0

    # If original hold_time/flight_time exist, compare and update when NaN or inconsistent
    if "data.hold_time" in df.columns or "hold_time" in df.columns:
        orig_hold = detect_columns(df, ["data.hold_time", "hold_time"])
        if orig_hold:
            df["hold_time"] = pd.to_numeric(df[orig_hold], errors="coerce")
            # Replace NaN or large discrepancy (>1ms) with calculated
            mask = df["hold_time"].isna() | (np.abs(df["hold_time"] - df["hold_time_ms_calc"]) > 1.0)
            df.loc[mask, "hold_time"] = df.loc[mask, "hold_time_ms_calc"]
    else:
        df["hold_time"] = df["hold_time_ms_calc"]

    if "data.flight_time" in df.columns or "flight_time" in df.columns:
        orig_flight = detect_columns(df, ["data.flight_time", "flight_time"])
        if orig_flight:
            df["flight_time"] = pd.to_numeric(df[orig_flight], errors="coerce")
            mask = df["flight_time"].isna() | (np.abs(df["flight_time"] - df["flight_time_ms_calc"]) > 1.0)
            df.loc[mask, "flight_time"] = df.loc[mask, "flight_time_ms_calc"]
    else:
        df["flight_time"] = df["flight_time_ms_calc"]

    return df


def main():
    db = get_db(URI, DB_NAME)
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    # Process mouse_events
    if "mouse_events" in db.list_collection_names():
        mdf = fetch_collection_as_df(db, "mouse_events")
        mdf_proc = process_mouse_events(mdf)
        out_mouse = out_dir / "mouse_events_metrics.csv"
        mdf_proc.to_csv(out_mouse, index=False)
        print(f"Mouse metrics salvo em: {out_mouse}")
    else:
        print("Coleção mouse_events não encontrada; pulando")

    # Process keystrokes
    if "keystrokes" in db.list_collection_names():
        kdf = fetch_collection_as_df(db, "keystrokes")
        kdf_proc = process_keystrokes(kdf)
        out_keys = out_dir / "keystrokes_metrics.csv"
        kdf_proc.to_csv(out_keys, index=False)
        print(f"Keystrokes metrics salvo em: {out_keys}")
    else:
        print("Coleção keystrokes não encontrada; pulando")


if __name__ == "__main__":
    main()
