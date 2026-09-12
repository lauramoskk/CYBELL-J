from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "experimentos_normalizados" / "comparativos"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _summarize_metric_row(metric_dict: dict, label: str, metric_name: str) -> dict:
    values = metric_dict.get(metric_name)
    if isinstance(values, dict):
        target = values.get("macro_f1")
        eer = values.get("eer")
    else:
        target = None
        eer = None

    return {
        "label": label,
        "metric": metric_name,
        "macro_f1": target,
        "eer": eer,
    }


def _collect_experiment_rows() -> list[dict]:
    rows: list[dict] = []

    for modality in ["mouse", "teclado"]:
        base = BASE_DIR / "experimentos_normalizados" / modality / "global"
        summary = _read_json(base / "summary.json")
        if not summary:
            continue

        metrics = summary.get("metrics", {})
        rows.append(
            {
                "modality": modality,
                "scenario": "global",
                "best_algorithm": summary.get("best_algorithm"),
                "macro_f1": metrics.get("macro_f1"),
                "eer": summary.get("eer", metrics.get("eer")),
                "unknown_threshold": summary.get("unknown_threshold"),
                "volumetria": "global",
                "sobreposicao": 75,
            }
        )

    return rows


def build_volumetria_table() -> pd.DataFrame:
    """Tabela comparando EER e F1 para cada volumetria testada."""
    rows = []

    volumetrias = [25, 30, 50, 100]
    for volume in volumetrias:
        rows.append(
            {
                "volumetria_eventos": volume,
                "eer": None,
                "f1_score_macro": None,
                "observacao": "não executado no pipeline atual",
            }
        )

    df = pd.DataFrame(rows)
    df["modelo_referencia"] = "pipeline atual"
    return df


def build_overlap_table() -> pd.DataFrame:
    """Tabela comparando sobreposições e o custo/benefício percebido."""
    rows = [
        {
            "sobreposicao": 25,
            "complexidade_computacional": "baixa",
            "seguranca_percebida": "moderada",
            "ganho_vs_custo": "baixo",
            "observacao": "base produzida com overlap menor; não há comparação final automatizada",
        },
        {
            "sobreposicao": 50,
            "complexidade_computacional": "média",
            "seguranca_percebida": "boa",
            "ganho_vs_custo": "médio",
            "observacao": " cenário intermediário; sem tabela consolidada",
        },
        {
            "sobreposicao": 75,
            "complexidade_computacional": "alta",
            "seguranca_percebida": "alta",
            "ganho_vs_custo": "alto",
            "observacao": "configuração atualmente usada no pipeline principal",
        },
    ]
    return pd.DataFrame(rows)


def build_latency_tradeoff_table() -> pd.DataFrame:
    """Curva de compromisso entre erro e latência."""
    latency_points = [5, 10, 15, 20]
    rows = []
    for latency in latency_points:
        rows.append(
            {
                "latencia_segundos": latency,
                "erro_estimado": None,
                "f1_estimado": None,
                "tradeoff": "não modelado no código atual",
            }
        )
    return pd.DataFrame(rows)


def save_summary_tables() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    volumetria_df = build_volumetria_table()
    overlap_df = build_overlap_table()
    latency_df = build_latency_tradeoff_table()

    volumetria_df.to_csv(OUTPUT_DIR / "comparativo_volumetria.csv", index=False)
    overlap_df.to_csv(OUTPUT_DIR / "comparativo_sobreposicao.csv", index=False)
    latency_df.to_csv(OUTPUT_DIR / "tradeoff_latencia.csv", index=False)

    with (OUTPUT_DIR / "comparativos.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "volumetria": volumetria_df.to_dict(orient="records"),
                "sobreposicao": overlap_df.to_dict(orient="records"),
                "latencia": latency_df.to_dict(orient="records"),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    save_summary_tables()
    print(f"Comparativos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
