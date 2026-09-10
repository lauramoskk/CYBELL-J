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

WINDOW_SIZES = [25, 30, 50, 100]

OVERLAP = 75

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("TECLADO — EXPERIMENTO DE VOLUMETRIA")
    print("=" * 80)

    print("Windows:", WINDOW_SIZES)
    print("Overlap:", f"{OVERLAP}%")
    print()

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

    print("Features:", len(features))
    print()

    # --------------------------------------------------------
    # Executar cada tamanho de janela
    # --------------------------------------------------------

    for window_size in WINDOW_SIZES:

        print("=" * 80)
        print(f"WINDOW SIZE: {window_size}")
        print(f"OVERLAP: {OVERLAP}%")
        print("=" * 80)

        df_exp = df[
            df["window_size"].eq(window_size)
            & df["requested_overlap_pct"].eq(OVERLAP)
        ].copy()

        output = (
            BASE_DIR
            / "experimentos_normalizados"
            / "teclado"
            / "volumetria"
            / f"window_{window_size}"
        )

        print(f"Janelas: {len(df_exp)}")
        print(f"Usuários: {df_exp['user'].nunique()}")
        print(f"Sessões: {df_exp['session_id'].nunique()}")
        print(f"Features: {len(features)}")
        print(f"Saída: {output}")
        print()

        # ----------------------------------------------------
        # Treinamento + avaliação
        # ----------------------------------------------------

        run_individual(
            df=df_exp,
            features=features,
            output_dir=output,
            modality="keyboard",
        )

        print()
        print(f"WINDOW {window_size} CONCLUÍDO")
        print()

    print("=" * 80)
    print("EXPERIMENTO DE VOLUMETRIA DO TECLADO CONCLUÍDO")
    print("=" * 80)


if __name__ == "__main__":
    main()