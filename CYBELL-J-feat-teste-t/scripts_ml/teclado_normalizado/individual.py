from pathlib import Path
import sys
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from scripts_ml.common.training import run_individual


DATASET = (
    BASE_DIR
    / "data"
    / "teclado_dataset"
    / "dataset_teclado.csv"
)

MANIFEST = (
    BASE_DIR
    / "data"
    / "teclado_dataset"
    / "dataset_teclado_manifest.json"
)

OUTPUT = (
    BASE_DIR
    / "experimentos_normalizados"
    / "teclado"
    / "individual"
)

WINDOW_SIZE = 50
OVERLAP = 75


def main():

    df = pd.read_csv(
        DATASET,
        low_memory=False,
    )

    with MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as file:

        features = json.load(file)[
            "feature_columns"
        ]

    df = df[
        df["window_size"].eq(WINDOW_SIZE)
        & df["requested_overlap_pct"].eq(OVERLAP)
    ].copy()

    print("=" * 80)
    print("TECLADO INDIVIDUAL NORMALIZADO")
    print("=" * 80)
    print("Features:", len(features))
    print("Saída:", OUTPUT)

    run_individual(
        df=df,
        features=features,
        output_dir=OUTPUT,
        modality="keyboard",
    )


if __name__ == "__main__":
    main()