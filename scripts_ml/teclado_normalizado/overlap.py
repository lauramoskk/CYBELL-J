from pathlib import Path
import sys
import json

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
    / "teclado_dataset"
    / "dataset_teclado.csv"
)

MANIFEST = (
    BASE_DIR
    / "data"
    / "teclado_dataset"
    / "dataset_teclado_manifest.json"
)

# Ponto 36 — Overlap
# Window fixo em 50 eventos
WINDOW_SIZE = 50

# Primeiro experimento: overlap de 25%
OVERLAP = 50

OUTPUT = (
    BASE_DIR
    / "experimentos_normalizados"
    / "teclado"
    / "overlap"
    / f"overlap_{OVERLAP}"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TECLADO — EXPERIMENTO DE OVERLAP")
    print("=" * 80)

    print(f"Window size: {WINDOW_SIZE}")
    print(f"Overlap: {OVERLAP}%")

    # --------------------------------------------------------
    # Carregar dataset
    # --------------------------------------------------------

    print("Carregando dataset...")

    df = pd.read_csv(
        DATASET,
        low_memory=False,
    )

    print("Dataset carregado.")
    print()

    # --------------------------------------------------------
    # Carregar features do manifesto
    # --------------------------------------------------------

    with MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as file:

        features = json.load(file)["feature_columns"]

    # --------------------------------------------------------
    # Filtrar configuração
    # --------------------------------------------------------

    df = df[
        df["window_size"].eq(WINDOW_SIZE)
        & df["requested_overlap_pct"].eq(OVERLAP)
    ].copy()

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
        modality="keyboard",
    )

    print()
    print("=" * 80)
    print("EXPERIMENTO CONCLUÍDO")
    print("=" * 80)


if __name__ == "__main__":
    main()
