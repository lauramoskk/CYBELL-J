from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from scripts_ml.common.training_multiclass import (
    run_multiclass,
)


DATASET = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)


OUTPUT = (
    BASE_DIR
    / "experimentos_finais_6_users"
    / "mouse"
    / "global"
)


USERS = [
    "Julia104soup",
    "bianca",
    "lala",
    "laura_oliveira",
    "lina",
    "yass",
]


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

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATASET}"
        )

    df = pd.read_csv(
        DATASET,
        low_memory=False,
    )

    df["user"] = (
        df["user"]
        .astype(str)
        .str.strip()
    )

    # Mantém mouse e trackpad.
    # O dataset já contém somente janelas
    # classificadas como alta confiança.
    df = df[
        df["window_size"].eq(WINDOW_SIZE)
        & df["requested_overlap_pct"].eq(OVERLAP)
    ].copy()

    features = [
        column
        for column in df.columns
        if column not in METADATA
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if len(features) != 89:
        raise RuntimeError(
            f"Esperadas 89 features, "
            f"encontradas {len(features)}."
        )

    print("=" * 80)
    print("CYBELL-J — MODELO FINAL DE PONTEIRO — 6 USUÁRIAS")
    print("=" * 80)

    print(
        "Mouse e trackpad tratados "
        "como uma única modalidade."
    )

    if "device_context" in df.columns:
        print("\nDistribuição por contexto:")
        print(
            df["device_context"]
            .value_counts()
            .to_string()
        )

    run_multiclass(
        df=df,
        features=features,
        users=USERS,
        output_dir=OUTPUT,
        modality="mouse",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
    )


if __name__ == "__main__":
    main()