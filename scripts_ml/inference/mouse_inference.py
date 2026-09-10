from pathlib import Path
import sys

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from scripts_ml.inference.mouse_features import extract_mouse_features


MODEL_PATH = (
    BASE_DIR
    / "experimentos_normalizados"
    / "mouse"
    / "global"
    / "model_global.joblib"
)


def carregar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def prever_mouse(eventos):
    """
    Recebe uma lista com eventos de mouse
    e retorna a previsão do modelo global.
    """

    if not eventos:
        raise ValueError(
            "Nenhum evento de mouse foi informado."
        )

    modelo = carregar_modelo()

    features_geradas = extract_mouse_features(eventos)

    features_esperadas = modelo["features"]

    # Garante exatamente as colunas usadas
    # durante o treinamento.
    features = features_geradas.reindex(
        columns=features_esperadas
    )

    pipeline = modelo["pipeline"]

    probabilidades = pipeline.predict_proba(
        features
    )[0]

    indice_previsto = probabilidades.argmax()

    usuario_previsto = (
        modelo["label_encoder"]
        .inverse_transform(
            [indice_previsto]
        )[0]
    )

    confianca = float(
        probabilidades[indice_previsto]
    )

    threshold = float(
        modelo["unknown_threshold"]
    )

    resultado = (
        "conhecido"
        if confianca >= threshold
        else "desconhecido"
    )

    return {
        "usuario_previsto": usuario_previsto,
        "confianca": confianca,
        "threshold": threshold,
        "resultado": resultado,
        "probabilidades": {
            usuario: float(probabilidade)
            for usuario, probabilidade in zip(
                modelo["label_encoder"].classes_,
                probabilidades,
            )
        },
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DE INFERÊNCIA DO MOUSE")
    print("=" * 60)

    modelo = carregar_modelo()

    print("Modelo:", MODEL_PATH)
    print("Algoritmo:", modelo["best_algorithm"])
    print("Usuários:", modelo["known_users"])
    print("Features:", len(modelo["features"]))
    print(
        "Threshold:",
        modelo["unknown_threshold"],
    )

    print("\n✅ Modelo carregado com sucesso.")