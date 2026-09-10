from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from werkzeug.security import check_password_hash
from mongo import keystrokes_collection, mouse_events_collection
from scripts_ml.inference.mouse_inference import prever_mouse
import time


# Cria o Blueprint para as rotas de API
api_bp = Blueprint("api", __name__)


@api_bp.route("/api/behavior", methods=["POST"])
@login_required
def receive_behavior_data():
    """
    Recebe dados comportamentais (teclado e mouse) do Front-end
    com validação de segurança
    ---
    tags:
      - Coleta de Dados
    responses:
      200:
        description: Dados validados e salvos com sucesso no MongoDB
      400:
        description: Dados inválidos ou vazios
      429:
        description: Excesso de volume de eventos (Proteção Anti-Flood)
    """

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Nenhum dado recebido"
        }), 400

    keyboard_data = data.get("keyboard", [])
    mouse_data = data.get("mouse", [])
    shortcuts_data = data.get("shortcuts", [])

    # 1. Validação de Janela Vazia (Confiança Degradada)
    if not keyboard_data and not mouse_data and not shortcuts_data:
        return jsonify({
            "status": "empty_window",
            "message": "Nenhum dado biométrico recebido nesta janela."
        }), 200

    # 2. Segurança: Proteção Anti-Flood / Anti-Bot
    # Limite de volume por lote
    if len(mouse_data) > 400 or len(keyboard_data) > 200:
        return jsonify({
            "status": "security_violation",
            "message": "Volume excessivo de eventos detectado."
        }), 429

    # 3. Sanitização e Vinculação Segura com o Usuário Logado
    current_user_id = current_user.username

    sanitized_keyboard = []

    for k in keyboard_data:
        sanitized_keyboard.append({
            "user": current_user_id,
            "key": str(k.get("key", ""))[:5],
            "event_type": str(k.get("event_type", "keydown")),
            "timestamp": int(k.get("timestamp", time.time()))
        })

    sanitized_mouse = []

    for m in mouse_data:
        x_val = float(m.get("x", 0.0))
        y_val = float(m.get("y", 0.0))

        # SEGURANÇA:
        # Descarta coordenadas fora de uma tela convencional
        # (ex: monitores de 4K até 3840x2160)
        # Impedindo valores espaciais absurdos ou negativos inválidos
        if 0 <= x_val <= 4000 and 0 <= y_val <= 3000:
            sanitized_mouse.append({
                "user": current_user_id,
                "event_type": str(m.get("event_type", "")),
                "device_type": str(m.get("device_type", "mouse")),
                "x": x_val,
                "y": y_val,
                "delta_x": m.get("delta_x", 0),
                "delta_y": m.get("delta_y", 0),
                "timestamp": int(m.get("timestamp", time.time()))
            })

    sanitized_shortcuts = []

    for s in shortcuts_data:
        hold_time = float(s.get("hold_time", 0.0))

        if hold_time >= 0:
            sanitized_shortcuts.append({
                "user": current_user_id,
                "key": str(s.get("key", "")),
                "event_type": "shortcut",
                "hold_time": hold_time,
                "timestamp": int(s.get("timestamp", time.time()))
            })

    # 4. Persistência Segura no MongoDB com Tratamento de Erros
    try:
        if sanitized_keyboard:
            keystrokes_collection.insert_many(sanitized_keyboard)

        if sanitized_mouse:
            mouse_events_collection.insert_many(sanitized_mouse)

        if sanitized_shortcuts:
            keystrokes_collection.insert_many(sanitized_shortcuts)

    except Exception as e:
        return jsonify({
            "error": "Erro ao persistir dados no banco NoSQL",
            "details": str(e)
        }), 500

    return jsonify({
        "status": "success",
        "message": "Dados biométricos validados e salvos no MongoDB"
    }), 200


@api_bp.route("/api/reauth", methods=["POST"])
@login_required
def reauth():
    """
    Reautenticação do usuário durante a sessão (confiança degradada)
    ---
    tags:
      - Autenticação
    parameters:
      - name: password
        in: formData
        type: string
        required: true
    responses:
      200:
        description: Senha correta, sessão restaurada
      400:
        description: Senha não enviada
      401:
        description: Senha incorreta
    """

    data = request.get_json(silent=True) or {}
    password = data.get("password", "")

    if not password:
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


@api_bp.route("/api/verify", methods=["POST"])
@login_required
def verify_ia():
    """
    Verificação da Inteligência Artificial
    ---
    tags:
      - Inteligência Artificial
    responses:
      200:
        description: Retorna o resultado da análise comportamental
      400:
        description: Dados insuficientes para análise
      500:
        description: Erro durante a análise
    """

    TAMANHO_JANELA = 50

    usuario_logado = current_user.username

    try:
        # Busca os eventos de mouse do usuário atualmente logado
        eventos = list(
            mouse_events_collection
            .find({"user": usuario_logado})
            .sort("timestamp", -1)
            .limit(TAMANHO_JANELA)
        )

        # Verifica se há eventos suficientes
        if len(eventos) < TAMANHO_JANELA:
            return jsonify({
                "status": "insufficient_data",
                "score": 0.0,
                "resultado": "indisponivel",
                "mensagem": (
                    f"Dados insuficientes para análise. "
                    f"São necessários pelo menos {TAMANHO_JANELA} "
                    f"eventos de mouse."
                ),
                "eventos_analisados": len(eventos)
            }), 400

        # Remove o identificador interno do MongoDB
        for evento in eventos:
            evento.pop("_id", None)

        # Executa a inferência real do modelo
        resultado_ia = prever_mouse(eventos)

        usuario_previsto = resultado_ia["usuario_previsto"]
        confianca = resultado_ia["confianca"]
        threshold = resultado_ia["threshold"]

        # O modelo considera conhecido somente quando
        # a confiança ultrapassa o threshold configurado.
        if resultado_ia["resultado"] == "conhecido":
            status_sessao = "legitimo"
        else:
            status_sessao = "suspeito"

        return jsonify({
            "status": "success",
            "score": confianca,
            "resultado": status_sessao,
            "usuario_previsto": usuario_previsto,
            "threshold": threshold,
            "probabilidades": resultado_ia["probabilidades"],
            "eventos_analisados": len(eventos),
            "mensagem": "Análise comportamental concluída"
        }), 200

    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Erro durante a análise comportamental.",
            "details": str(e)
        }), 500