from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from scripts_ml.common.training import run_global


DATASET = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)

OUTPUT = (
    BASE_DIR
    / "experimentos_normalizados"
    / "mouse"
    / "global"
)

WINDOW_SIZE = 50
OVERLAP = 75


METADATA = {
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
    "device_mouse_pct",
    "device_trackpad_pct",
    "device_context",
    "device_confidence",
    "device_high_confidence",
    "device_label_source",
    "target_user_device",
}


def main():

    df = pd.read_csv(
        DATASET,
        low_memory=False,
    )

    df["user"] = (
        df["user"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["device_context"]
        .astype(str)
        .str.lower()
        .eq("mouse")
        & df["window_size"].eq(WINDOW_SIZE)
        & df["requested_overlap_pct"].eq(OVERLAP)
    ].copy()

    features = [
        column
        for column in df.columns
        if column not in METADATA
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    print("=" * 80)
    print("MOUSE GLOBAL NORMALIZADO")
    print("=" * 80)
    print("Classes conhecidas: Yass e Lina")
    print("Features:", len(features))
    print("Saída:", OUTPUT)

    run_global(
        df=df,
        features=features,
        output_dir=OUTPUT,
        modality="mouse",
    )


if __name__ == "__main__":
    main()