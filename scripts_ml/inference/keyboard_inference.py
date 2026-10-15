from pathlib import Path
import sys

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts_ml.inference.keyboard_features import (
    extract_keyboard_features,
)


MODEL_PATH = (
    BASE_DIR
    / "treinamento_f"
    / "experimentos_finais_6_users"
    / "teclado"
    / "global"
    / "model_global_6_users.joblib"
)

_modelo_cache = None


def carregar_modelo():
    """Carrega e valida o modelo global de teclado."""

    global _modelo_cache

    if _modelo_cache is not None:
        return _modelo_cache

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo de teclado não encontrado: {MODEL_PATH}"
        )

    modelo = joblib.load(MODEL_PATH)

    if not isinstance(modelo, dict):
        raise ValueError(
            "O arquivo do modelo de teclado é inválido."
        )

    campos_obrigatorios = {
        "pipeline",
        "features",
        "label_encoder",
        "classes",
        "n_features",
    }

    campos_ausentes = campos_obrigatorios - set(modelo.keys())

    if campos_ausentes:
        raise ValueError(
            "Campos ausentes no modelo: "
            + ", ".join(sorted(campos_ausentes))
        )

    if len(modelo["features"]) != int(modelo["n_features"]):
        raise ValueError(
            "A quantidade de características não corresponde "
            "ao informado no modelo."
        )

    _modelo_cache = modelo
    return _modelo_cache


def prever_teclado(eventos):
    """
    Prevê a classe mais provável para uma janela de eventos
    de teclado.

    A previsão não confirma, por si só, a identidade da pessoa.
    """

    if not isinstance(eventos, (list, tuple)) or not eventos:
        raise ValueError(
            "Nenhum evento de teclado foi informado."
        )

    modelo = carregar_modelo()

    features_geradas = extract_keyboard_features(eventos)
    features_esperadas = modelo["features"]

    faltantes = [
        nome
        for nome in features_esperadas
        if nome not in features_geradas.columns
    ]

    if faltantes:
        raise ValueError(
            "O extrator não gerou todas as características "
            "esperadas pelo modelo: "
            + ", ".join(faltantes)
        )

    # Usa as mesmas características e a mesma ordem do treinamento.
    features = features_geradas.reindex(
        columns=features_esperadas
    )

    pipeline = modelo["pipeline"]
    label_encoder = modelo["label_encoder"]

    probabilidades = pipeline.predict_proba(features)[0]

    classes_pipeline = getattr(pipeline, "classes_", None)

    if classes_pipeline is not None:
        classes_previstas = label_encoder.inverse_transform(
            classes_pipeline
        )
    else:
        classes_previstas = modelo["classes"]

    classes_previstas = [
        str(classe)
        for classe in classes_previstas
    ]

    if len(classes_previstas) != len(probabilidades):
        raise ValueError(
            "A quantidade de classes não corresponde "
            "às probabilidades retornadas."
        )

    indice = int(probabilidades.argmax())

    usuario_previsto = classes_previstas[indice]
    confianca = float(probabilidades[indice])

    probabilidades_por_usuario = {
        usuario: float(probabilidade)
        for usuario, probabilidade in zip(
            classes_previstas,
            probabilidades,
        )
    }

    return {
        "usuario_previsto": usuario_previsto,
        "confianca": confianca,
        "probabilidades": probabilidades_por_usuario,
    }


if __name__ == "__main__":
    try:
        modelo = carregar_modelo()

        print("Modelo de teclado carregado!")
        print("Caminho:", MODEL_PATH)
        print("Algoritmo:", modelo["best_algorithm"])
        print("Características:", modelo["n_features"])
        print("Janela:", modelo["window_size"])
        print("Usuários:", modelo["classes"])

    except Exception as erro:
        print("Erro ao carregar o modelo:", erro)
        raise