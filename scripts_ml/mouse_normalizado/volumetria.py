from pathlib import Path
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts_ml.common.training import run_individual


DATASET = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)

WINDOWS = [25, 30, 50, 100]
OVERLAP = 75

BASE_OUTPUT = (
    BASE_DIR
    / "experimentos_normalizados"
    / "mouse"
    / "volumetria"
)


def main():
    print("=" * 80)
    print("MOUSE — EXPERIMENTO DE VOLUMETRIA")
    print("=" * 80)

    print("Carregando dataset...")
    df = pd.read_csv(DATASET, low_memory=False)
    print("Dataset carregado.")
    print()

    metadata_columns = {
        "user",
        "session_id",
        "window_size",
        "requested_overlap_pct",
        "start_time",
        "end_time",
        "timestamp",
        "device",
        "modality",
    }

    features = [
        column
        for column in df.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    print(f"Features: {len(features)}")
    print(f"Overlap fixado: {OVERLAP}%")
    print()

    for window_size in WINDOWS:

        print("=" * 80)
        print(f"WINDOW SIZE: {window_size}")
        print("=" * 80)

        df_window = df[
            (df["window_size"] == window_size)
            & (df["requested_overlap_pct"] == OVERLAP)
        ].copy()

        output = BASE_OUTPUT / f"window_{window_size}"

        print(f"Janelas: {len(df_window)}")
        print(f"Usuários: {df_window['user'].nunique()}")
        print(f"Sessões: {df_window['session_id'].nunique()}")
        print(f"Saída: {output}")
        print()

        run_individual(
            df=df_window,
            features=features,
            output_dir=output,
            modality="mouse",
        )

        print()

    print("=" * 80)
    print("EXPERIMENTO DE VOLUMETRIA CONCLUÍDO")
    print("=" * 80)


if __name__ == "__main__":
    main()