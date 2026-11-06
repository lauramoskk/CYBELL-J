from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

USERS = {
    "Julia104soup",
    "bianca",
    "lala",
    "laura_oliveira",
    "lina",
    "yass",
}

MODELS = {
    "mouse": {
        "path": (
            BASE_DIR
            / "experimentos_finais_6_users"
            / "mouse"
            / "global"
            / "model_global_6_users.joblib"
        ),
        "dataset": (
            BASE_DIR
            / "data"
            / "mouse_dataset"
            / "dataset_device_high_confidence.csv"
        ),
        "n_features": 89,
    },
    "keyboard": {
        "path": (
            BASE_DIR
            / "experimentos_finais_6_users"
            / "teclado"
            / "global"
            / "model_global_6_users.joblib"
        ),
        "dataset": (
            BASE_DIR
            / "data"
            / "teclado_dataset"
            / "dataset_teclado.csv"
        ),
        "n_features": 59,
    },
}


def validate(name, config):

    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    bundle = joblib.load(
        config["path"]
    )

    pipeline = bundle["pipeline"]
    features = bundle["features"]
    classes = set(
        bundle["classes"]
    )

    assert classes == USERS
    assert len(features) == config["n_features"]

    assert list(
        pipeline.named_steps
    ) == [
        "imputer",
        "scaler",
        "model",
    ]

    df = pd.read_csv(
        config["dataset"],
        low_memory=False,
    )

    row = df[features].iloc[
        [0]
    ]

    proba = pipeline.predict_proba(
        row
    )

    assert proba.shape == (
        1,
        6,
    )

    assert np.isclose(
        proba.sum(),
        1.0,
    )

    print("Classes:", sorted(classes))
    print("Features:", len(features))
    print(
        "Algoritmo:",
        bundle["best_algorithm"],
    )
    print(
        "Pipeline:",
        list(
            pipeline.named_steps
        ),
    )
    print(
        "Probabilidades:",
        np.round(
            proba[0],
            4,
        ),
    )
    print("STATUS: OK")


def main():

    for name, config in (
        MODELS.items()
    ):
        validate(
            name,
            config,
        )

    print(
        "\nTODOS OS MODELOS "
        "PASSARAM NA VALIDAÇÃO."
    )


if __name__ == "__main__":
    main()