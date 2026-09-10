from pathlib import Path
import sys
import pandas as pd

# ============================================================
# CONFIGURAÇÃO
# ============================================================

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

# Ponto 36 — Overlap
# Window fixo em 50 eventos
WINDOW_SIZE = 50

# Primeiro experimento: overlap de 25%
OVERLAP = 50

OUTPUT = (
    BASE_DIR
    / "experimentos_normalizados"
    / "mouse"
    / "overlap"
    / f"overlap_{OVERLAP}"
)

# ============================================================
# METADADOS
# ============================================================

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
    "duration_s",
    "n_events",
    "device_mouse_pct",
    "device_trackpad_pct",
    "device_context",
    "device_confidence",
    "device_high_confidence",
    "device_label_source",
    "target_user_device",
}

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MOUSE — EXPERIMENTO DE OVERLAP")
    print("=" * 80)

    print(f"Window size: {WINDOW_SIZE}")
    print(f"Overlap: {OVERLAP}%")

    # --------------------------------------------------------
    # Carregar dataset
    # --------------------------------------------------------

    print("Carregando dataset...")

    df = pd.read_csv(DATASET)

    df["user"] = df["target_user"]

    print("Dataset carregado.")
    print()

    # --------------------------------------------------------
    # Filtrar configuração
    # --------------------------------------------------------

    df = df[
        (df["device_context"] == "mouse")
        & (df["window_size"] == WINDOW_SIZE)
        & (df["requested_overlap_pct"] == OVERLAP)
    ].copy()

    # --------------------------------------------------------
    # Identificar features
    # --------------------------------------------------------

    features = [
        column
        for column in df.columns
        if column not in METADATA
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    # --------------------------------------------------------
    # Informações do experimento
    # --------------------------------------------------------

    print(f"Janelas: {len(df)}")
    print(f"Usuários: {df['user'].nunique()}")
    print(f"Sessões: {df['session_id'].nunique()}")
    print(f"Features: {len(features)}")
    print(f"Saída: {OUTPUT}")
    print()

    # --------------------------------------------------------
    # Treinamento + avaliação
    # --------------------------------------------------------

    run_individual(
        df=df,
        features=features,
        output_dir=OUTPUT,
        modality="mouse",
    )

    print()
    print("=" * 80)
    print("EXPERIMENTO CONCLUÍDO")
    print("=" * 80)


if __name__ == "__main__":
    main()