from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[2]
BASE_OUTPUT = BASE_DIR / "experimentos_normalizados" / "teclado" / "volumetria"

WINDOWS = [25, 30, 50, 100]


def carregar_resultado(window, usuario):
    arquivo = (
        BASE_OUTPUT
        / f"window_{window}"
        / f"{usuario}_summary.json"
    )

    with arquivo.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 90)
    print("TABELA FINAL — EXPERIMENTO DE VOLUMETRIA — TECLADO")
    print("=" * 90)

    print(
        f"{'Window':<10}"
        f"{'Yass EER':>15}"
        f"{'Yass F1':>15}"
        f"{'Lina EER':>15}"
        f"{'Lina F1':>15}"
    )

    print("-" * 90)

    for window in WINDOWS:
        yass = carregar_resultado(window, "yass")
        lina = carregar_resultado(window, "lina")

        yass_eer = yass["eer"] * 100
        yass_f1 = yass["metrics"]["macro_f1"] * 100

        lina_eer = lina["eer"] * 100
        lina_f1 = lina["metrics"]["macro_f1"] * 100

        print(
            f"{window:<10}"
            f"{yass_eer:>14.2f}%"
            f"{yass_f1:>14.2f}%"
            f"{lina_eer:>14.2f}%"
            f"{lina_f1:>14.2f}%"
        )

    print("=" * 90)
    print("RESULTADOS LIDOS DOS ARQUIVOS GERADOS PELO EXPERIMENTO")
    print("=" * 90)


if __name__ == "__main__":
    main()