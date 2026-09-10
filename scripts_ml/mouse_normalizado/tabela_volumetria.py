from pathlib import Path
import json

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VOLUMETRIA_DIR = (
    BASE_DIR
    / "experimentos_normalizados"
    / "mouse"
    / "volumetria"
)

WINDOWS = [25, 30, 50, 100]


# ============================================================
# LEITURA DOS RESULTADOS
# ============================================================

resultados = []

for window in WINDOWS:

    pasta = VOLUMETRIA_DIR / f"window_{window}"

    yass_file = pasta / "yass_summary.json"
    lina_file = pasta / "lina_summary.json"

    if not yass_file.exists() or not lina_file.exists():
        print(f"ERRO: resultados não encontrados para window {window}")
        continue

    with open(yass_file, "r", encoding="utf-8") as arquivo:
        yass = json.load(arquivo)

    with open(lina_file, "r", encoding="utf-8") as arquivo:
        lina = json.load(arquivo)

    resultados.append({
        "window": window,
        "yass": yass,
        "lina": lina
    })


# ============================================================
# TABELA
# ============================================================

print()
print("=" * 90)
print("TABELA FINAL — EXPERIMENTO DE VOLUMETRIA")
print("=" * 90)

print(
    f"{'Window':<10}"
    f"{'Yass EER':<15}"
    f"{'Yass F1':<15}"
    f"{'Lina EER':<15}"
    f"{'Lina F1':<15}"
)

print("-" * 90)

for resultado in resultados:

    window = resultado["window"]
    yass = resultado["yass"]
    lina = resultado["lina"]

    yass_eer = yass["eer"] * 100
    yass_f1 = yass["metrics"]["macro_f1"] * 100

    lina_eer = lina["eer"] * 100
    lina_f1 = lina["metrics"]["macro_f1"] * 100

    print(
        f"{window:<10}"
        f"{yass_eer:>8.2f}%       "
        f"{yass_f1:>8.2f}%       "
        f"{lina_eer:>8.2f}%       "
        f"{lina_f1:>8.2f}%"
    )

print("=" * 90)
print("RESULTADOS LIDOS DOS ARQUIVOS GERADOS PELO EXPERIMENTO")
print("=" * 90)