from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

OUT = (
    BASE_DIR
    / "experimentos_normalizados"
    / "evaluation_unknown"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


WINDOW = 50
OVERLAP = 75


UNKNOWN_USERS = [
    "Julia104soup",
    "bianca",
    "lala",
    "laura_oliveira",
]


MOUSE_DATA = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)

KEYBOARD_DATA = (
    BASE_DIR
    / "data"
    / "teclado_dataset"
    / "dataset_teclado.csv"
)


def model_path(
    modality,
    experiment,
    name,
):

    return (
        BASE_DIR
        / "experimentos_normalizados"
        / modality
        / experiment
        / name
    )


def load_mouse():

    df = pd.read_csv(
        MOUSE_DATA,
        low_memory=False,
    )

    df["user"] = (
        df["user"]
        .astype(str)
        .str.strip()
    )

    return df[
        df["device_context"]
        .astype(str)
        .str.lower()
        .eq("mouse")
        & df["window_size"].eq(WINDOW)
        & df["requested_overlap_pct"].eq(OVERLAP)
    ].copy()


def load_keyboard():

    df = pd.read_csv(
        KEYBOARD_DATA,
        low_memory=False,
    )

    df["user"] = (
        df["user"]
        .astype(str)
        .str.strip()
    )

    return df[
        df["window_size"].eq(WINDOW)
        & df["requested_overlap_pct"].eq(OVERLAP)
    ].copy()


def global_unknown(
    df,
    modality,
):

    bundle = joblib.load(
        model_path(
            modality,
            "global",
            "model_global.joblib",
        )
    )

    model = bundle["pipeline"]
    features = bundle["features"]
    encoder = bundle["label_encoder"]

    threshold = float(
        bundle["unknown_threshold"]
    )

    test = df[
        df["user"].isin(
            UNKNOWN_USERS
        )
    ].copy()

    probabilities = (
        model.predict_proba(
            test[features]
        )
    )

    predicted_index = (
        probabilities.argmax(
            axis=1
        )
    )

    predicted_user = (
        encoder.inverse_transform(
            predicted_index
        )
    )

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    rejected = (
        confidence
        < threshold
    )

    return pd.DataFrame(
        {
            "modality":
                modality,

            "actual_user":
                test["user"].to_numpy(),

            "predicted_known_user":
                predicted_user,

            "confidence":
                confidence,

            "threshold":
                threshold,

            "decision":
                np.where(
                    rejected,
                    "UNKNOWN",
                    "KNOWN",
                ),

            "correct_rejection":
                rejected,
        }
    )


def cross_individual(
    df,
    modality,
    target,
    unknown_user,
):

    bundle = joblib.load(
        model_path(
            modality,
            "individual",
            f"{target}.joblib",
        )
    )

    if (
        bundle[
            "excluded_unknown_user"
        ]
        != unknown_user
    ):
        raise RuntimeError(
            "Usuária testada não é "
            "a desconhecida reservada."
        )

    test = df[
        df["user"]
        .eq(unknown_user)
    ].copy()

    score = (
        bundle["pipeline"]
        .predict_proba(
            test[
                bundle["features"]
            ]
        )[:, 1]
    )

    threshold = float(
        bundle["threshold"]
    )

    rejected = (
        score < threshold
    )

    return pd.DataFrame(
        {
            "modality":
                modality,

            "target_model":
                target,

            "unknown_user":
                unknown_user,

            "score_as_target":
                score,

            "threshold":
                threshold,

            "decision":
                np.where(
                    rejected,
                    "REJECT",
                    "ACCEPT",
                ),

            "correct_rejection":
                rejected,
        }
    )


def summarize_global(
    df,
):

    result = (
        df.groupby(
            [
                "modality",
                "actual_user",
            ],
            as_index=False,
        )
        .agg(
            windows=(
                "actual_user",
                "size",
            ),

            rejection_rate=(
                "correct_rejection",
                "mean",
            ),

            mean_confidence=(
                "confidence",
                "mean",
            ),
        )
    )

    result[
        "false_acceptance_rate"
    ] = (
        1
        - result[
            "rejection_rate"
        ]
    )

    return result


def summarize_cross(
    df,
):

    result = (
        df.groupby(
            [
                "modality",
                "target_model",
                "unknown_user",
            ],
            as_index=False,
        )
        .agg(
            windows=(
                "unknown_user",
                "size",
            ),

            rejection_rate=(
                "correct_rejection",
                "mean",
            ),

            mean_score=(
                "score_as_target",
                "mean",
            ),
        )
    )

    result[
        "false_acceptance_rate"
    ] = (
        1
        - result[
            "rejection_rate"
        ]
    )

    return result


def main():

    print("=" * 80)
    print(
        "CYBELL-J — TESTES "
        "COM USUÁRIOS DESCONHECIDOS"
    )
    print("=" * 80)

    mouse = load_mouse()
    keyboard = load_keyboard()

    global_results = pd.concat(
        [
            global_unknown(
                mouse,
                "mouse",
            ),

            global_unknown(
                keyboard,
                "teclado",
            ),
        ],
        ignore_index=True,
    )

    cross_results = pd.concat(
        [
            cross_individual(
                mouse,
                "mouse",
                "yass",
                "lina",
            ),

            cross_individual(
                mouse,
                "mouse",
                "lina",
                "yass",
            ),

            cross_individual(
                keyboard,
                "teclado",
                "yass",
                "lina",
            ),

            cross_individual(
                keyboard,
                "teclado",
                "lina",
                "yass",
            ),
        ],
        ignore_index=True,
    )

    global_summary = (
        summarize_global(
            global_results
        )
    )

    cross_summary = (
        summarize_cross(
            cross_results
        )
    )

    global_results.to_csv(
        OUT
        / "global_unknown_windows.csv",
        index=False,
    )

    global_summary.to_csv(
        OUT
        / "global_unknown_summary.csv",
        index=False,
    )

    cross_results.to_csv(
        OUT
        / "individual_cross_windows.csv",
        index=False,
    )

    cross_summary.to_csv(
        OUT
        / "individual_cross_summary.csv",
        index=False,
    )

    print(
        "\nGLOBAL — PESSOAS "
        "DESCONHECIDAS"
    )

    print(
        global_summary.to_string(
            index=False
        )
    )

    print(
        "\nINDIVIDUAL — "
        "YASS x LINA"
    )

    print(
        cross_summary.to_string(
            index=False
        )
    )

    print(
        "\nResultados:"
    )

    print(
        OUT
    )


if __name__ == "__main__":
    main()