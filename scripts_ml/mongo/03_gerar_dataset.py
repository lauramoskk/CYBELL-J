"""03_gerar_dataset.py

Consolida features de `keystrokes` e `mouse_events` por `user_id` e `session_id`
e gera `data/dataset.csv` pronto para treinar modelos de ML (uma linha por sessão).

Uso: python 03_gerar_dataset.py
"""
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)

def infer_user_id(kdf: pd.DataFrame, mdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.concat([kdf["user_id"].dropna(), mdf["user_id"].dropna()]).drop_duplicates()
    if len(users) == 1:
        user = users.iloc[0]
        kdf["user_id"] = kdf["user_id"].fillna(user)
        mdf["user_id"] = mdf["user_id"].fillna(user)
    return kdf, mdf


def read_metrics():
    keys_path = DATA_DIR / "keystrokes_metrics.csv"
    mouse_path = DATA_DIR / "mouse_events_metrics.csv"

    if not keys_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {keys_path}. Rode 01_extrair_dados.py/02_calcular_metricas.py primeiro.")
    if not mouse_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {mouse_path}. Rode 01_extrair_dados.py/02_calcular_metricas.py primeiro.")

    kdf = pd.read_csv(keys_path)
    mdf = pd.read_csv(mouse_path)
    return kdf, mdf


def build_keyboard_features(kdf: pd.DataFrame) -> pd.DataFrame:
    # Garantir colunas necessárias
    if "hold_time" not in kdf.columns or "flight_time" not in kdf.columns:
        raise RuntimeError("Keystrokes precisa conter `hold_time` e `flight_time`")

    keyboard_features = (
        kdf
        .groupby(["user_id", "session_id"], dropna=False)
        .agg(
            hold_mean=("hold_time", "mean"),
            hold_std=("hold_time", "std"),
            hold_min=("hold_time", "min"),
            hold_max=("hold_time", "max"),
            hold_median=("hold_time", "median"),
            hold_p25=("hold_time", lambda x: x.quantile(0.25)),
            hold_p75=("hold_time", lambda x: x.quantile(0.75)),
            hold_p90=("hold_time", lambda x: x.quantile(0.90)),

            flight_mean=("flight_time", "mean"),
            flight_std=("flight_time", "std"),
            flight_min=("flight_time", "min"),
            flight_max=("flight_time", "max"),
            flight_median=("flight_time", "median"),
            flight_p25=("flight_time", lambda x: x.quantile(0.25)),
            flight_p75=("flight_time", lambda x: x.quantile(0.75)),
            flight_p90=("flight_time", lambda x: x.quantile(0.90)),

            n_keys=("hold_time", "count"),
        )
        .reset_index()
    )
    return keyboard_features


def build_mouse_features(mdf: pd.DataFrame) -> pd.DataFrame:
    # Espera-se colunas: speed, acceleration, distance
    required = ["speed", "acceleration", "distance"]
    for c in required:
        if c not in mdf.columns:
            raise RuntimeError(f"mouse_events precisa conter a coluna '{c}'")

    mouse_features = (
        mdf
        .groupby(["user_id", "session_id"], dropna=False)
        .agg(
            speed_mean=("speed", "mean"),
            speed_std=("speed", "std"),
            speed_max=("speed", "max"),
            speed_median=("speed", "median"),
            speed_p25=("speed", lambda x: x.quantile(0.25)),
            speed_p75=("speed", lambda x: x.quantile(0.75)),
            speed_p90=("speed", lambda x: x.quantile(0.90)),

            accel_mean=("acceleration", "mean"),
            accel_std=("acceleration", "std"),
            accel_max=("acceleration", "max"),
            accel_median=("acceleration", "median"),
            accel_p25=("acceleration", lambda x: x.quantile(0.25)),
            accel_p75=("acceleration", lambda x: x.quantile(0.75)),
            accel_p90=("acceleration", lambda x: x.quantile(0.90)),

            total_distance=("distance", "sum"),

            n_mouse_events=("distance", "count"),
        )
        .reset_index()
    )
    return mouse_features


def consolidate_and_save(kf: pd.DataFrame, mf: pd.DataFrame, out_path: Path):
    dataset = kf.merge(mf, on=["user_id", "session_id"], how="inner")

    # Preencher NaNs com 0 para features ausentes em casos inesperados
    dataset = dataset.fillna(0)

    # Reordenar colunas para formato desejado
    cols_order = [
        "user_id",
        "session_id",
        "hold_mean",
        "hold_std",
        "hold_min",
        "hold_max",
        "hold_median",
        "hold_p25",
        "hold_p75",
        "hold_p90",

        "flight_mean",
        "flight_std",
        "flight_min",
        "flight_max",
        "flight_median",
        "flight_p25",
        "flight_p75",
        "flight_p90",

        "speed_mean",
        "speed_std",
        "speed_median",
        "speed_p25",
        "speed_p75",
        "speed_p90",
        "speed_max",

        "accel_mean",
        "accel_std",
        "accel_median",
        "accel_p25",
        "accel_p75",
        "accel_p90",
        "accel_max",

        "total_distance",
        "n_keys",
        "n_mouse_events",
    ]

    # Garantir que todas as colunas existam antes de reordenar
    for c in cols_order:
        if c not in dataset.columns:
            dataset[c] = 0

    dataset = dataset[cols_order]

    dataset.to_csv(out_path, index=False)
    print(f"Dataset salvo em: {out_path} — linhas: {len(dataset)}")


def main():
    kdf, mdf = read_metrics()
    kdf, mdf = infer_user_id(kdf, mdf)

    before_k = len(kdf)
    before_m = len(mdf)
    kdf = kdf.dropna(subset=["session_id"])
    mdf = mdf.dropna(subset=["session_id"])
    kdf = kdf[kdf["hold_time"].notna() | kdf["flight_time"].notna()]
    mdf = mdf[mdf["distance"].notna()]
    print(f"Keystrokes: linhas antes={before_k} depois={len(kdf)}")
    print(f"Mouse: linhas antes={before_m} depois={len(mdf)}")

    kf = build_keyboard_features(kdf)
    mf = build_mouse_features(mdf)
    out_path = DATA_DIR / "dataset.csv"
    consolidate_and_save(kf, mf, out_path)


if __name__ == "__main__":
    main()
