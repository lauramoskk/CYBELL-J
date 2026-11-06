from pathlib import Path
import sys
import json

import pandas as pd


BASE_DIR = Path(
    __file__
).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BASE_DIR),
    )


from scripts_ml.common.training_multiclass_smote import (
    run_multiclass_smote,
)


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
    / "experimentos_finais_6_users_smote"
    / "teclado"
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


def main():

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: "
            f"{DATASET}"
        )

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest não encontrado: "
            f"{MANIFEST}"
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

    with MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(
            file
        )

    features = manifest[
        "feature_columns"
    ]

    if len(features) != 59:
        raise RuntimeError(
            f"Esperadas 59 features "
            f"de teclado, encontradas "
            f"{len(features)}."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Features ausentes no CSV: "
            + ", ".join(
                missing_features
            )
        )

    df = df[
        df["window_size"].eq(
            WINDOW_SIZE
        )
        & df[
            "requested_overlap_pct"
        ].eq(
            OVERLAP
        )
    ].copy()

    print("=" * 80)

    print(
        "CYBELL-J — TECLADO "
        "COM SMOTE — 6 USUÁRIAS"
    )

    print("=" * 80)

    run_multiclass_smote(
        df=df,
        features=features,
        users=USERS,
        output_dir=OUTPUT,
        modality="keyboard",
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
    )


if __name__ == "__main__":
    main()