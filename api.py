from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.security import check_password_hash

from mongo import keystrokes_collection, mouse_events_collection
from scripts_ml.inference.mouse_inference import prever_mouse

import time


# Blueprint das rotas da API
api_bp = Blueprint("api", __name__)


# ============================================
# Recebimento de dados comportamentais
# ============================================

@api_bp.route("/api/behavior", methods=["POST"])
@login_required
def receive_behavior_data():
    """Recebe, valida e armazena eventos comportamentais."""

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Nenhum dado válido foi recebido."
        }), 400

    keyboard_data = data.get("keyboard", [])
    mouse_data = data.get("mouse", [])
    shortcuts_data = data.get("shortcuts", [])

    # Valida os tipos recebidos
    if not all(
        isinstance(items, list)
        for items in (
            keyboard_data,
            mouse_data,
            shortcuts_data
        )
    ):
        return jsonify({
            "status": "error",
            "message": "Os eventos devem ser enviados em listas."
        }), 400

    # Janela sem eventos
    if not keyboard_data and not mouse_data and not shortcuts_data:
        return jsonify({
            "status": "empty_window",
            "message": (
                "Nenhum dado comportamental recebido nesta janela."
            )
        }), 200

    # Proteção de volume por requisição
    print(
        f"Eventos recebidos — mouse: {len(mouse_data)}, "
        f"teclado: {len(keyboard_data)}"
    )

    if len(mouse_data) > 400 or len(keyboard_data) > 200:
        return jsonify({
            "status": "security_violation",
            "message": "Volume excessivo de eventos detectado."
        }), 429

    # O usuário vem da sessão autenticada,
    # nunca do JSON enviado pelo navegador.
    username = current_user.username

    sanitized_keyboard = []
    sanitized_mouse = []
    sanitized_shortcuts = []

    try:
        # Sanitização dos eventos de teclado
        for event in keyboard_data:
            if not isinstance(event, dict):
                continue

            sanitized_keyboard.append({
                "user": username,
                "key": str(event.get("key", ""))[:5],
                "event_type": str(
                    event.get("event_type", "keydown")
                )[:30],
                "timestamp": int(
                    event.get("timestamp", time.time())
                )
            })

        # Sanitização dos eventos de mouse
        for event in mouse_data:
            if not isinstance(event, dict):
                continue

            x_val = float(event.get("x", 0.0))
            y_val = float(event.get("y", 0.0))

            if not (
                0 <= x_val <= 4000
                and 0 <= y_val <= 3000
            ):
                continue

            sanitized_mouse.append({
                "user": username,
                "event_type": str(
                    event.get("event_type", "")
                )[:30],
                "device_type": str(
                    event.get("device_type", "mouse")
                )[:20],
                "x": x_val,
                "y": y_val,
                "delta_x": float(event.get("delta_x", 0)),
                "delta_y": float(event.get("delta_y", 0)),
                "timestamp": int(
                    event.get("timestamp", time.time())
                )
            })

        # Atalhos são armazenados na coleção de teclado
        for event in shortcuts_data:
            if not isinstance(event, dict):
                continue

            hold_time = float(event.get("hold_time", 0.0))

            if hold_time < 0:
                continue

            sanitized_shortcuts.append({
                "user": username,
                "key": str(event.get("key", ""))[:20],
                "event_type": "shortcut",
                "hold_time": hold_time,
                "timestamp": int(
                    event.get("timestamp", time.time())
                )
            })

        # Persistência no MongoDB
        if sanitized_keyboard:
            keystrokes_collection.insert_many(sanitized_keyboard)

        if sanitized_mouse:
            mouse_events_collection.insert_many(sanitized_mouse)

        if sanitized_shortcuts:
            keystrokes_collection.insert_many(sanitized_shortcuts)

    except (TypeError, ValueError, OverflowError):
        return jsonify({
            "status": "error",
            "message": (
                "Um ou mais eventos contêm valores inválidos."
            )
        }), 400

    except Exception:
        current_app.logger.exception(
            "Erro ao persistir eventos comportamentais."
        )

        return jsonify({
            "status": "error",
            "message": "Erro ao salvar os dados comportamentais."
        }), 500

    return jsonify({
        "status": "success",
        "message": "Dados comportamentais salvos no MongoDB.",
        "eventos_salvos": {
            "teclado": len(sanitized_keyboard),
            "mouse": len(sanitized_mouse),
            "atalhos": len(sanitized_shortcuts)
        }
    }), 200


# ============================================
# Reautenticação por senha
# ============================================

@api_bp.route("/api/reauth", methods=["POST"])
@login_required
def reauth():
    """Confirma a senha do usuário autenticado."""

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Dados de reautenticação inválidos."
        }), 400

    password = data.get("password", "")

    if not isinstance(password, str) or not password:
        return jsonify({
            "status": "error",
            "message": "Senha não informada."
        }), 400

    if not check_password_hash(current_user.password, password):
        return jsonify({
            "status": "invalid",
            "message": "Senha incorreta."
        }), 401

    return jsonify({
        "status": "success",
        "message": "Reautenticação confirmada."
    }), 200


# ============================================
# Verificação comportamental pela IA
# ============================================

@api_bp.route("/api/verify", methods=["POST"])
@login_required
def verify_ia():
    """
    Analisa os eventos de mouse associados ao usuário autenticado.

    A previsão do modelo não é considerada prova definitiva
    da identidade. Os critérios utilizados nesta versão são
    provisórios e precisam ser validados experimentalmente.
    """

    tamanho_janela = 50
    usuario_logado = current_user.username

    try:
        # Busca somente eventos associados à conta autenticada.
        eventos = list(
            mouse_events_collection
            .find({"user": usuario_logado})
            .sort("timestamp", -1)
            .limit(tamanho_janela)
        )

        # Verifica se existem dados suficientes
        if len(eventos) < tamanho_janela:
            return jsonify({
                "status": "insufficient_data",
                "score": 0.0,
                "resultado": "indisponivel",
                "mensagem": (
                    f"Dados insuficientes. São necessários "
                    f"{tamanho_janela} eventos de mouse."
                ),
                "eventos_analisados": len(eventos)
            }), 400

        # Remove o identificador interno do MongoDB
        for evento in eventos:
            evento.pop("_id", None)

        # Executa a inferência
        resultado_ia = prever_mouse(eventos)

        usuario_previsto = str(
            resultado_ia["usuario_previsto"]
        )

        confianca = float(
            resultado_ia["confianca"]
        )

        probabilidades = {
            str(usuario): float(probabilidade)
            for usuario, probabilidade
            in resultado_ia["probabilidades"].items()
        }

        # Normaliza os nomes para a comparação
        usuario_previsto_normalizado = (
            usuario_previsto.strip().casefold()
        )

        usuario_logado_normalizado = (
            str(usuario_logado).strip().casefold()
        )

        # Ordena as probabilidades para calcular
        # a diferença entre a primeira e a segunda classe.
        probabilidades_ordenadas = sorted(
            probabilidades.values(),
            reverse=True
        )

        maior_probabilidade = (
            probabilidades_ordenadas[0]
            if probabilidades_ordenadas
            else 0.0
        )

        segunda_probabilidade = (
            probabilidades_ordenadas[1]
            if len(probabilidades_ordenadas) > 1
            else 0.0
        )

        margem = (
            maior_probabilidade - segunda_probabilidade
        )

        # Critério provisório:
        # confiança mínima de 80% e margem mínima de 20 pontos percentuais.
        #
        # Estes valores são heurísticos e não estão validados.
        # Não devem ser tratados como garantia de identidade.

        if (
            usuario_previsto_normalizado
            == usuario_logado_normalizado
        ):
            resultado = "inconclusivo"

            mensagem = (
                "O modelo previu o usuário autenticado, mas ainda "
                "não existe um critério de confiança validado "
                "para confirmar a identidade."
            )

        elif (
            confianca >= 0.80
            and margem >= 0.20
        ):
            resultado = "suspeito"

            mensagem = (
                "O modelo previu uma classe diferente da conta "
                "autenticada, com confiança e margem elevadas "
                "segundo um critério provisório. A divergência "
                "precisa ser avaliada e não comprova, por si só, "
                "que outra pessoa esteja utilizando a conta."
            )

        else:
            resultado = "inconclusivo"

            mensagem = (
                "O modelo previu uma classe diferente da conta "
                "autenticada, mas a confiança ou a margem entre "
                "as duas maiores probabilidades não atende ao "
                "critério provisório. Não é possível concluir "
                "que exista uma divergência real de identidade."
            )

        return jsonify({
            "status": "success",
            "score": confianca,
            "resultado": resultado,
            "usuario_previsto": usuario_previsto,
            "probabilidades": probabilidades,
            "eventos_analisados": len(eventos),
            "mensagem": mensagem
        }), 200

    except ValueError as error:
        current_app.logger.warning(
            "Erro de valor durante a verificação comportamental: %s",
            error
        )

        return jsonify({
            "status": "error",
            "message": (
                "Não foi possível processar os dados "
                "da verificação."
            )
        }), 400

    except Exception:
        current_app.logger.exception(
            "Erro durante a verificação de identidade."
        )

        return jsonify({
            "status": "error",
            "message": "Erro durante a análise comportamental."
        }), 500