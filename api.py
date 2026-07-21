from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from mongo import keystrokes_collection, mouse_events_collection
import time

# Cria o Blueprint para as rotas de API
api_bp = Blueprint("api", __name__)

@api_bp.route("/api/behavior", methods=["POST"])
@login_required
def receive_behavior_data():
    """
    Recebe dados comportamentais (teclado e mouse) do Front-end com validação de segurança
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
        return jsonify({"error": "Nenhum dado recebido"}), 400

    keyboard_data = data.get("keyboard", [])
    mouse_data = data.get("mouse", [])
    shortcuts_data = data.get("shortcuts", [])

    # 1. Validação de Janela Vazia (Confiança Degradada)
    if not keyboard_data and not mouse_data and not shortcuts_data:
        return jsonify({
            "status": "empty_window", 
            "message": "Nenhum dado biométrico recebido nesta janela."
        }), 200

    # 2. Segurança: Proteção Anti-Flood / Anti-Bot (Limite de volume por lote)
    if len(mouse_data) > 400 or len(keyboard_data) > 200:
        return jsonify({
            "status": "security_violation",
            "message": "Volume excessivo de eventos detectado."
        }), 429

    # 3. Sanitização e Vinculación Segura com o Usuário Logado
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
        
        # SEGURANÇA: Descarta coordenadas fora de uma tela convencional (ex: monitores de 4K até 3840x2160)
        # Impedindo injeção de valores espaciais absurdos ou negativos inválidos
        if 0 <= x_val <= 4000 and 0 <= y_val <= 3000:
            sanitized_mouse.append({
                "user": current_user_id,
                "event_type": str(m.get("event_type", "")),
                "x": x_val,
                "y": y_val,
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
        return jsonify({"error": "Erro ao persistir dados no banco NoSQL", "details": str(e)}), 500

    return jsonify({
        "status": "success", 
        "message": "Dados biométricos validados e salvos no MongoDB"
    }), 200

@api_bp.route("/api/verify", methods=["POST"])
@login_required
def verify_ia():
    """
    Verificação da Inteligência Artificial (Esqueleto/Placeholder)
    ---
    tags:
      - Inteligência Artificial
    responses:
      200:
        description: Retorna o score de legitimidade calculado pela IA
    """
    fake_score = 0.92
    limiar_seguranca = 0.80

    status_sessao = "legitimo" if fake_score >= limiar_seguranca else "suspeito"

    return jsonify({
        "status": "success",
        "score": fake_score,
        "resultado": status_sessao,
        "mensagem": "Análise comportamental concluída"
    }), 200