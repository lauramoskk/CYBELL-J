from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd

from scripts_ml.common.training import (
    nested_cv,
    calculate_eer,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

LATENCIAS = [5, 10, 15, 20]

WINDOW_SIZE = 50
OVERLAP = 75

USUARIOS = ["yass", "lina"]

DATASETS = {
    "mouse": (
        BASE_DIR
        / "data"
        / "mouse_dataset"
        / "dataset_device_high_confidence.csv"
    ),
    "teclado": (
        BASE_DIR
        / "data"
        / "teclado_dataset"
        / "dataset_teclado.csv"
    ),
}

MODELS = {
    "mouse": (
        BASE_DIR
        / "experimentos_normalizados"
        / "mouse"
        / "volumetria"
        / "window_50"
    ),
    "teclado": (
        BASE_DIR
        / "experimentos_normalizados"
        / "teclado"
        / "volumetria"
        / "window_50"
    ),
}

OUTPUT_DIR = (
    BASE_DIR
    / "experimentos_normalizados"
    / "tradeoff_latencia"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURAÇÃO DOS MODELOS
# ============================================================

IMPOSTORS = [
    "Julia104soup",
    "bianca",
    "lala",
    "laura_oliveira",
]

CONFIGS = {
    "yass": {
        "algorithm": "RandomForest",
        "excluded": "lina",
    },
    "lina": {
        "algorithm": "XGBoost",
        "excluded": "yass",
    },
}


# ============================================================
# FEATURES
# ============================================================

def get_features_from_model(
    modality,
    target_user,
):
    """
    Usa exatamente as features armazenadas
    no modelo utilizado nos experimentos
    anteriores.
    """

    model_path = (
        MODELS[modality]
        / f"{target_user}.joblib"
    )

    bundle = joblib.load(
        model_path
    )

    return bundle["features"]


# ============================================================
# OOF + LATÊNCIA
# ============================================================

def calculate_latency_results(
    subset,
    oof,
    target_user,
    latency,
):
    """
    Seleciona as janelas OOF disponíveis
    nos primeiros N segundos de cada sessão.

    Depois calcula uma probabilidade média
    por sessão e calcula o EER.
    """

    meta = subset[
        [
            "session_id",
            "window_start_ms",
            "user",
        ]
    ].copy()

    # Relaciona cada previsão OOF
    # à janela original.
    oof = oof.copy()

    oof["session_id"] = (
        meta.iloc[
            oof["row_index"]
        ]["session_id"]
        .to_numpy()
    )

    oof["window_start_ms"] = (
        meta.iloc[
            oof["row_index"]
        ]["window_start_ms"]
        .to_numpy()
    )

    oof["user"] = (
        meta.iloc[
            oof["row_index"]
        ]["user"]
        .to_numpy()
    )

    # Primeiro instante observado
    # de cada sessão.
    session_start = (
        oof.groupby(
            "session_id"
        )["window_start_ms"]
        .transform("min")
    )

    elapsed_s = (
        oof["window_start_ms"]
        - session_start
    ) / 1000.0

    oof["elapsed_s"] = elapsed_s

    # Apenas janelas disponíveis
    # dentro da latência.
    selected = oof[
        oof["elapsed_s"] <= latency
    ].copy()

    if selected.empty:
        return None

    # Uma decisão por sessão:
    # média das probabilidades das
    # janelas disponíveis.
    sessions = (
        selected
        .groupby("session_id")
        .agg(
            probability=(
                "probability",
                "mean",
            ),
            user=(
                "user",
                "first",
            ),
            n_windows=(
                "probability",
                "count",
            ),
        )
        .reset_index()
    )

    sessions["y_true"] = (
        sessions["user"]
        .eq(target_user)
        .astype(int)
    )

    # Verificação para garantir
    # presença das duas classes.
    if sessions["y_true"].nunique() < 2:
        return None

    eer, threshold = calculate_eer(
    sessions["y_true"].to_numpy(),
    sessions["probability"].to_numpy(),
)

    return {
        "latency_s": latency,
        "target_user": target_user,
        "eer": eer,
        "sessions": len(sessions),
        "mean_windows": float(
            sessions["n_windows"].mean()
        ),
    }


# ============================================================
# EXPERIMENTO POR MODALIDADE
# ============================================================

def run_modality(
    modality,
):

    print()
    print("=" * 80)
    print(
        f"{modality.upper()} - "
        "TRADE-OFF DE LATÊNCIA"
    )
    print("=" * 80)

    dataset = DATASETS[
        modality
    ]

    print(
        f"Dataset: {dataset}"
    )

    df = pd.read_csv(
        dataset,
        low_memory=False,
    )

    # --------------------------------------------------------
    # 50 eventos / 75% overlap
    # --------------------------------------------------------

    df = df[
        (df["window_size"] == WINDOW_SIZE)
        & (
            df["requested_overlap_pct"]
            == OVERLAP
        )
    ].copy()

    df = df.reset_index(
        drop=True
    )

    print(
        f"Janelas 50/75: {len(df)}"
    )

    print(
        f"Sessões: "
        f"{df['session_id'].nunique()}"
    )

    all_results = []

    # --------------------------------------------------------
    # YASS / LINA
    # --------------------------------------------------------

    for target_user in USUARIOS:

        config = CONFIGS[
            target_user
        ]

        print()
        print("=" * 80)
        print(
            f"ALVO: {target_user}"
        )
        print(
            f"Algoritmo: "
            f"{config['algorithm']}"
        )
        print("=" * 80)

        allowed_users = (
            [target_user]
            + IMPOSTORS
        )

        subset = df[
            df["user"].isin(
                allowed_users
            )
        ].copy()

        subset = subset.reset_index(
            drop=True
        )

        # O usuário cruzado não participa.
        assert (
            config["excluded"]
            not in subset["user"].unique()
        )

        # Usa exatamente as features
        # do modelo correspondente
        # aos experimentos 35/36.
        features = get_features_from_model(
            modality,
            target_user,
        )

        print(
            f"Features: {len(features)}"
        )

        X = subset[
            features
        ]

        y = (
            subset["user"]
            .eq(target_user)
            .astype(int)
            .to_numpy()
        )

        groups = (
            subset[
                "session_id"
            ].astype(str)
        )

        print(
            "Executando nested CV "
            "para gerar probabilidades OOF..."
        )

        folds, oof, search = nested_cv(
            X,
            y,
            groups,
            config["algorithm"],
        )

        print(
            "Nested CV concluído."
        )

        print(
            f"Melhores parâmetros: "
            f"{search.best_params_}"
        )

        # ----------------------------------------------------
        # Latências
        # ----------------------------------------------------

        for latency in LATENCIAS:

            result = calculate_latency_results(
                subset=subset,
                oof=oof,
                target_user=target_user,
                latency=latency,
            )

            if result is None:
                print(
    f"DEBUG {target_user} {latency}s:",
    type(result),
    result,
)
                continue

            all_results.append(
                {
                    "modality": modality,
                    **result,
                }
            )

            print(
                f"Latência: {latency:>2}s | "
                f"Sessões: "
                f"{result['sessions']:>2} | "
                f"Janelas/sessão: "
                f"{result['mean_windows']:.2f} | "
                f"EER: "
                f"{result['eer']:.4f} "
                f"({result['eer'] * 100:.2f}%)"
            )

    # --------------------------------------------------------
    # Salvar resultados
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    output_csv = (
        OUTPUT_DIR
        / f"{modality}_resultados.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    print()
    print(
        f"Resultados salvos em:"
    )
    print(output_csv)

    return results_df


# ============================================================
# GRÁFICO
# ============================================================

def generate_graph(
    results_df,
    modality,
):

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(9, 6)
    )

    for target_user in USUARIOS:

        data = results_df[
            results_df["target_user"]
            == target_user
        ].sort_values(
            "latency_s"
        )

        plt.plot(
            data["latency_s"],
            data["eer"] * 100,
            marker="o",
            label=target_user,
        )

    plt.xlabel(
        "Latência de observação (s)"
    )

    plt.ylabel(
        "EER (%)"
    )

    plt.title(
        "Trade-off entre latência e erro - "
        f"{modality.capitalize()}"
    )

    plt.xticks(
        LATENCIAS
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_png = (
        OUTPUT_DIR
        / f"{modality}_tradeoff.png"
    )

    plt.savefig(
        output_png,
        dpi=300,
    )

    plt.close()

    print(
        f"Gráfico salvo em:"
    )
    print(output_png)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "CYBELL-J - EXPERIMENTO 37"
    )
    print(
        "TRADE-OFF ENTRE LATÊNCIA E ERRO"
    )
    print("=" * 80)

    mouse = run_modality(
        "mouse"
    )

    generate_graph(
        mouse,
        "mouse",
    )

    teclado = run_modality(
        "teclado"
    )

    generate_graph(
        teclado,
        "teclado",
    )

    final = pd.concat(
        [
            mouse,
            teclado,
        ],
        ignore_index=True,
    )

    final = final[
        [
            "modality",
            "latency_s",
            "target_user",
            "eer",
            "sessions",
            "mean_windows",
        ]
    ]

    output_final = (
        OUTPUT_DIR
        / "tabela_tradeoff_final.csv"
    )

    final.to_csv(
        output_final,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "EXPERIMENTO 37 CONCLUÍDO"
    )
    print("=" * 80)

    print()
    print(
        final.to_string(
            index=False
        )
    )

    print()
    print(
        f"Tabela final: "
        f"{output_final}"
    )


if __name__ == "__main__":
    main()