from pathlib import Path
import sys

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts_ml.inference.mouse_features import extract_mouse_features


# Modelo global treinado para os seis usuários.
MODEL_PATH = (
    BASE_DIR
    / "treinamento_f"
    / "experimentos_finais_6_users"
    / "mouse"
    / "global"
    / "model_global_6_users.joblib"
)

# Mantém o modelo carregado em memória após a primeira utilização.
_modelo_cache = None


def carregar_modelo():
    """Carrega e valida a estrutura básica do modelo."""

    global _modelo_cache

    if _modelo_cache is not None:
        return _modelo_cache

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}"
        )

    modelo = joblib.load(MODEL_PATH)

    if not isinstance(modelo, dict):
        raise ValueError(
            "O arquivo do modelo não contém um dicionário válido."
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
            "O artefato do modelo não contém os campos necessários: "
            + ", ".join(sorted(campos_ausentes))
        )

    features = modelo["features"]

    if not isinstance(features, list) or not features:
        raise ValueError(
            "A lista de características do modelo é inválida."
        )

    if len(features) != int(modelo["n_features"]):
        raise ValueError(
            "A quantidade de características não corresponde "
            "ao informado no artefato."
        )

    _modelo_cache = modelo

    return _modelo_cache


def prever_mouse(eventos):
    """
    Analisa uma janela de eventos de mouse.

    Retorna o usuário previsto, a confiança da previsão
    e as probabilidades por classe.

    A previsão não confirma, por si só, a identidade
    da pessoa que está utilizando a sessão.
    """

    if not isinstance(eventos, (list, tuple)) or not eventos:
        raise ValueError(
            "Nenhum evento de mouse foi informado."
        )

    modelo = carregar_modelo()

    # Extrai as características dos eventos.
    features_geradas = extract_mouse_features(eventos)

    features_esperadas = modelo["features"]

    # Não permite que características ausentes sejam preenchidas
    # silenciosamente com valores vazios.
    features_ausentes = [
        nome
        for nome in features_esperadas
        if nome not in features_geradas.columns
    ]

    if features_ausentes:
        raise ValueError(
            "O extrator não gerou todas as características "
            "esperadas pelo modelo. Ausentes: "
            + ", ".join(features_ausentes)
        )

    # Mantém exatamente a ordem utilizada no treinamento.
    features = features_geradas.reindex(
        columns=features_esperadas
    )

    pipeline = modelo["pipeline"]
    label_encoder = modelo["label_encoder"]

    probabilidades = pipeline.predict_proba(features)[0]

    # Identifica as classes na mesma ordem das colunas
    # retornadas por predict_proba.
    classes_pipeline = getattr(pipeline, "classes_", None)

    if classes_pipeline is not None:
        try:
            classes_previstas = label_encoder.inverse_transform(
                classes_pipeline
            )
        except (TypeError, ValueError):
            classes_previstas = modelo["classes"]
    else:
        classes_previstas = modelo["classes"]

    classes_previstas = [
        str(classe)
        for classe in classes_previstas
    ]

    if len(classes_previstas) != len(probabilidades):
        raise ValueError(
            "A quantidade de classes não corresponde "
            "à quantidade de probabilidades retornadas."
        )

    indice_previsto = int(probabilidades.argmax())

    usuario_previsto = classes_previstas[indice_previsto]

    confianca = float(probabilidades[indice_previsto])

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
    print("=" * 60)
    print("TESTE DE INFERÊNCIA DO MOUSE")
    print("=" * 60)

    try:
        modelo = carregar_modelo()

        print("Modelo:", MODEL_PATH)
        print("Algoritmo:", modelo["best_algorithm"])
        print("Usuários:", modelo["classes"])
        print("Características:", len(modelo["features"]))
        print("Janela de treinamento:", modelo["window_size"])
        print("\nModelo carregado com sucesso.")

    except Exception as erro:
        print("Falha ao carregar o modelo:", erro)
        raise