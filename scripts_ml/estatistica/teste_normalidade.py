"""Testes de normalidade para features de mouse e trackpad.

Executar a partir da raiz do projeto:

    python scripts_ml/estatistica/teste_normalidade.py

As janelas sao agregadas por usuario e sessao antes dos testes. Isso evita
tratar janelas sobrepostas como observacoes independentes.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import normaltest, shapiro, ttest_rel, wilcoxon


BASE_DIR = Path(__file__).resolve().parents[2]
MOUSE_PATH = (
    BASE_DIR
    / "data"
    / "mouse_dataset"
    / "dataset_device_high_confidence.csv"
)
TRACKPAD_PATH = (
    BASE_DIR
    / "data"
    / "trackpad_dataset"
    / "dataset_trackpad_high_confidence.csv"
)
RAW_METRICS_PATH = BASE_DIR / "data" / "mouse_events_metrics.csv"
OUTPUT_DIR = BASE_DIR / "experimentos_normalizados" / "estatistica"
OUTPUT_CSV = OUTPUT_DIR / "teste_normalidade.csv"
OUTPUT_JSON = OUTPUT_DIR / "teste_normalidade.json"
OUTPUT_DESCRIPTIVE = OUTPUT_DIR / "estatisticas_descritivas_mouse_trackpad.csv"
OUTPUT_CV = OUTPUT_DIR / "comparacao_cv_mouse_trackpad.csv"

WINDOW_SIZE = 50
OVERLAP = 75
ALPHA = 0.05
MIN_SAMPLES = 3
SHAPIRO_MAX_SAMPLES = 5000

METADATA = {
    "_id",
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
    "n_events",
    "device_mouse_pct",
    "device_trackpad_pct",
    "device_context",
    "device_confidence",
    "device_high_confidence",
    "device_label_source",
    "target_user_device",
    "event_type",
    "device_type",
    "user_id",
    "user_key",
    "timestamp",
    "_t",
    "time_gap",
    "new_block",
    "session_block",
    "missing_session",
    "prev_valid_session",
    "_x",
    "_y",
}


def load_modality(path: Path, modality: str) -> pd.DataFrame:
    if not path.exists() and RAW_METRICS_PATH.exists():
        path = RAW_METRICS_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Base de {modality} nao encontrada: {path}\n"
            "Execute primeiro o pipeline de geracao dos datasets."
        )

    df = pd.read_csv(path, low_memory=False)
    required = {"user", "session_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Base de {modality} sem colunas obrigatorias: "
            f"{', '.join(sorted(missing))}"
        )

    modality_columns = {"device_context", "device_type"}
    if not modality_columns.intersection(df.columns):
        raise ValueError(
            f"Base de {modality} sem coluna de modalidade. "
            "Informe um dataset separado de mouse/trackpad com "
            "device_context ou device_type."
        )

    if "device_context" in df.columns:
        df = df[
            df["device_context"].astype(str).str.lower().eq(modality)
        ].copy()
    elif "device_type" in df.columns:
        df = df[
            df["device_type"].astype(str).str.lower().eq(modality)
        ].copy()

    if "window_size" in df.columns:
        df = df[df["window_size"].eq(WINDOW_SIZE)].copy()
    if "requested_overlap_pct" in df.columns:
        df = df[df["requested_overlap_pct"].eq(OVERLAP)].copy()

    df["modality"] = modality
    return df


def session_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        column
        for column in df.columns
        if column not in METADATA | {"modality"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not features:
        raise ValueError("Nenhuma feature numerica foi encontrada na base.")

    grouped = (
        df.groupby(["user", "session_id"], dropna=False)[features]
        .mean()
        .reset_index()
    )
    return grouped


def feature_names(*dataframes: pd.DataFrame) -> list[str]:
    shared = set(dataframes[0].columns)
    for dataframe in dataframes[1:]:
        shared.intersection_update(dataframe.columns)
    return sorted(
        column
        for column in shared
        if column not in {"user", "session_id"}
        and column not in METADATA
        and pd.api.types.is_numeric_dtype(dataframes[0][column])
    )


def descriptive_statistics(
    mouse: pd.DataFrame,
    trackpad: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for modality, dataframe in (("mouse", mouse), ("trackpad", trackpad)):
        for feature in features:
            values = pd.to_numeric(dataframe[feature], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            mean = float(values.mean()) if len(values) else None
            std = float(values.std(ddof=1)) if len(values) > 1 else None
            cv = (
                float(std / abs(mean) * 100)
                if mean is not None and std is not None and mean != 0
                else None
            )
            rows.append(
                {
                    "modalidade": modality,
                    "feature": feature,
                    "n_sessoes": int(len(values)),
                    "media": mean,
                    "desvio_padrao": std,
                    "coeficiente_variacao_pct": cv,
                }
            )
    return pd.DataFrame(rows)


def compare_cv(descriptive: pd.DataFrame) -> pd.DataFrame:
    pivot = descriptive.pivot(
        index="feature",
        columns="modalidade",
        values="coeficiente_variacao_pct",
    ).reset_index()
    pivot = pivot.rename(columns={"mouse": "mouse_cv_pct", "trackpad": "trackpad_cv_pct"})
    pivot["diferenca_trackpad_menos_mouse_pct"] = (
        pivot["trackpad_cv_pct"] - pivot["mouse_cv_pct"]
    )
    pivot["menor_variabilidade"] = np.select(
        [
            pivot["mouse_cv_pct"].isna() | pivot["trackpad_cv_pct"].isna(),
            pivot["mouse_cv_pct"] < pivot["trackpad_cv_pct"],
            pivot["trackpad_cv_pct"] < pivot["mouse_cv_pct"],
        ],
        ["indisponivel", "mouse", "trackpad"],
        default="igual",
    )
    return pivot


def run_tests(values: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    n = int(values.size)

    result: dict[str, object] = {
        "n_sessoes": n,
        "shapiro_stat": None,
        "shapiro_p": None,
        "shapiro_normal": None,
        "dagostino_stat": None,
        "dagostino_p": None,
        "dagostino_normal": None,
        "observacao": None,
    }

    if n < MIN_SAMPLES:
        result["observacao"] = "Amostra insuficiente; minimo de 3 sessoes."
        return result

    shapiro_values = values
    if n > SHAPIRO_MAX_SAMPLES:
        shapiro_values = values.sample(SHAPIRO_MAX_SAMPLES, random_state=42)

    shapiro_stat, shapiro_p = shapiro(shapiro_values)
    result.update(
        {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "shapiro_normal": bool(shapiro_p >= ALPHA),
        }
    )

    if n >= 8:
        dagostino_stat, dagostino_p = normaltest(values)
        result.update(
            {
                "dagostino_stat": float(dagostino_stat),
                "dagostino_p": float(dagostino_p),
                "dagostino_normal": bool(dagostino_p >= ALPHA),
            }
        )
    else:
        result["observacao"] = (
            "D'Agostino-Pearson requer pelo menos 8 sessoes."
        )

    return result


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    valid = [
        (index, value)
        for index, value in enumerate(p_values)
        if value is not None and np.isfinite(value)
    ]
    adjusted: list[float | None] = [None] * len(p_values)
    previous = 1.0

    for rank, (index, value) in enumerate(
        sorted(valid, key=lambda item: item[1], reverse=True),
        start=1,
    ):
        adjusted_value = min(previous, value * len(valid) / (len(valid) - rank + 1))
        adjusted[index] = float(adjusted_value)
        previous = adjusted_value

    return adjusted


def run_hypothesis_tests(
    mouse: pd.DataFrame,
    trackpad: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["user", "session_id"]
    paired = mouse.merge(
        trackpad,
        on=keys,
        how="inner",
        suffixes=("_mouse", "_trackpad"),
    )
    features = feature_names(mouse, trackpad)
    rows: list[dict[str, object]] = []

    for feature in features:
        mouse_values = pd.to_numeric(
            paired[f"{feature}_mouse"], errors="coerce"
        )
        trackpad_values = pd.to_numeric(
            paired[f"{feature}_trackpad"], errors="coerce"
        )
        valid = pd.DataFrame(
            {"mouse": mouse_values, "trackpad": trackpad_values}
        ).replace([np.inf, -np.inf], np.nan).dropna()
        differences = valid["trackpad"] - valid["mouse"]
        n = len(valid)

        row: dict[str, object] = {
            "feature": feature,
            "n_pares": n,
            "mouse_media": None,
            "trackpad_media": None,
            "diferenca_media_trackpad_menos_mouse": None,
            "diferenca_mediana_trackpad_menos_mouse": None,
            "t_pareado_stat": None,
            "t_pareado_p": None,
            "wilcoxon_stat": None,
            "wilcoxon_p": None,
            "cohen_dz": None,
            "hipotese_rejeitada_ajustada": None,
            "observacao": None,
        }

        if n < 2:
            row["observacao"] = "Pares insuficientes; minimo de 2."
            rows.append(row)
            continue

        row.update(
            {
                "mouse_media": float(valid["mouse"].mean()),
                "trackpad_media": float(valid["trackpad"].mean()),
                "diferenca_media_trackpad_menos_mouse": float(differences.mean()),
                "diferenca_mediana_trackpad_menos_mouse": float(differences.median()),
            }
        )

        if differences.std(ddof=1) > 0:
            t_stat, t_p = ttest_rel(valid["trackpad"], valid["mouse"])
            row["t_pareado_stat"] = float(t_stat)
            row["t_pareado_p"] = float(t_p)
            row["cohen_dz"] = float(differences.mean() / differences.std(ddof=1))

        if not np.allclose(differences.to_numpy(), 0):
            wilcoxon_stat, wilcoxon_p = wilcoxon(
                valid["trackpad"], valid["mouse"],
                alternative="two-sided",
            )
            row["wilcoxon_stat"] = float(wilcoxon_stat)
            row["wilcoxon_p"] = float(wilcoxon_p)
        else:
            row["observacao"] = "Todas as diferencas pareadas sao zero."

        rows.append(row)

    result = pd.DataFrame(rows)
    result["t_pareado_p_ajustado"] = benjamini_hochberg(
        result["t_pareado_p"].tolist()
    )
    result["wilcoxon_p_ajustado"] = benjamini_hochberg(
        result["wilcoxon_p"].tolist()
    )
    result["hipotese_rejeitada_ajustada"] = (
        result["wilcoxon_p_ajustado"].fillna(
            result["t_pareado_p_ajustado"]
        ) < ALPHA
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testa a normalidade das features por sessao."
    )
    parser.add_argument(
        "--mouse",
        type=Path,
        default=MOUSE_PATH,
        help="Caminho do CSV de mouse.",
    )
    parser.add_argument(
        "--trackpad",
        type=Path,
        default=TRACKPAD_PATH,
        help="Caminho do CSV de trackpad.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Tamanho da janela a analisar.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help="Percentual de sobreposicao da janela.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    used_raw_fallback = (
        not args.mouse.exists() or not args.trackpad.exists()
    ) and RAW_METRICS_PATH.exists()

    global WINDOW_SIZE, OVERLAP
    WINDOW_SIZE = args.window_size
    OVERLAP = args.overlap

    try:
        mouse = session_features(
            load_modality(args.mouse, "mouse")
        )
        trackpad = session_features(
            load_modality(args.trackpad, "trackpad")
        )
    except FileNotFoundError as error:
        raise SystemExit(
            f"{error}\n\n"
            "Os CSVs brutos nao estao versionados neste checkout. "
            "Gere-os com scripts_ml/trackpad/01_analisar_trackpad.py "
            "e o pipeline de mouse, ou informe os caminhos com "
            "--mouse CAMINHO e --trackpad CAMINHO."
        ) from error

    rows: list[dict[str, object]] = []
    for modality, data in (("mouse", mouse), ("trackpad", trackpad)):
        features = feature_names(data, data)
        for feature in features:
            row = {
                "modalidade": modality,
                "feature": feature,
            }
            row.update(run_tests(data[feature]))
            rows.append(row)

    result = pd.DataFrame(rows)
    hypothesis = run_hypothesis_tests(mouse, trackpad)
    descriptive = descriptive_statistics(mouse, trackpad, feature_names(mouse, trackpad))
    cv_comparison = compare_cv(descriptive)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    descriptive.to_csv(OUTPUT_DESCRIPTIVE, index=False, encoding="utf-8-sig")
    cv_comparison.to_csv(OUTPUT_CV, index=False, encoding="utf-8-sig")
    hypothesis.to_csv(
        OUTPUT_DIR / "testes_hipotese_mouse_trackpad.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "alpha": ALPHA,
        "window_size": WINDOW_SIZE,
        "overlap_pct": OVERLAP,
        "fonte": (
            str(RAW_METRICS_PATH.relative_to(BASE_DIR))
            if used_raw_fallback
            else "datasets de janelas informados"
        ),
        "fallback_metricas_brutas": used_raw_fallback,
        "filtro_window_overlap_aplicado": not used_raw_fallback,
        "unidade_observacao": "usuario + sessao",
        "mouse_sessoes": int(len(mouse)),
        "trackpad_sessoes": int(len(trackpad)),
        "pares_mouse_trackpad": int(
            len(mouse.merge(trackpad, on=["user", "session_id"], how="inner"))
        ),
        "testes_hipotese": [
            "t pareado",
            "Wilcoxon pareado",
            "Benjamini-Hochberg",
        ],
        "interpretacao": (
            "p_valor >= alpha nao rejeita normalidade; nao prova que a "
            "distribuicao seja normal."
        ),
    }
    OUTPUT_JSON.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Resultado CSV: {OUTPUT_CSV}")
    print(
        "Testes de hipotese: "
        f"{OUTPUT_DIR / 'testes_hipotese_mouse_trackpad.csv'}"
    )
    print(f"Metadados JSON: {OUTPUT_JSON}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()