from flask import Blueprint, request, jsonify
from flask_login import login_required
from mongo import keystrokes_collection, mouse_events_collection

# Cria o Blueprint para as rotas de API
api_bp = Blueprint("api", __name__)

# ==========================================
# ROTAS DA API BACKEND (INTEGRANTE 2)
# ==========================================

@api_bp.route("/api/behavior", methods=["POST"])
@login_required
def receive_behavior_data():
    """
    Recebe dados comportamentais (teclado, mouse e atalhos) do Front-end
    ---
    tags:
      - Coleta de Dados
    responses:
      200:
        description: Dados salvos com sucesso no MongoDB
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Nenhum dado recebido"}), 400

    keyboard_data = data.get("keyboard", [])
    mouse_data = data.get("mouse", [])
    shortcuts_data = data.get("shortcuts", [])
    validated_shortcuts = []

    for shortcut in shortcuts_data:
        # Garante que o atalho tem os campos obrigatórios e que o hold_time é positivo (evita manipulação no front)
        if "key" in shortcut and "timestamp" in shortcut:
            hold_time = shortcut.get("hold_time", 0)
            if hold_time >= 0:  # O tempo que o Ctrl ficou pressionado não pode ser negativo
                validated_shortcuts.append({
                    "key": shortcut.get("key"),
                    "event_type": "shortcut",
                    "hold_time": hold_time,
                    "timestamp": shortcut.get("timestamp")
                })

    # 1. LÓGICA DE CONFIANÇA DEGRADADA (Janela Vazia)
    if len(keyboard_data) == 0 and len(mouse_data) == 0 and len(shortcuts_data) == 0:
        return jsonify({
            "status": "empty_window", 
            "message": "Nenhum dado biométrico recebido nesta janela."
        }), 200

    # 2. REGRA ANTI-BOT BÁSICA (Evitar injeção de scripts)
    if len(mouse_data) > 10:
        first_time = mouse_data[0].get("timestamp")
        # Se todos os eventos de mouse tiverem exatamente o mesmo milissegundo, é um bot
        all_same_time = all(m.get("timestamp") == first_time for m in mouse_data)
        if all_same_time:
            return jsonify({
                "status": "bot_detected",
                "message": "Atividade sintética detectada."
            }), 403

    # 3. COMPENSAÇÃO DINÂMICA DE PESOS
    keyboard_weight = 0.5
    mouse_weight = 0.5
    shortcut_weight = 0.0

    # Cenário A: Leitura/Navegação (Muito mouse, pouco teclado)
    if len(keyboard_data) < 5 and len(mouse_data) > 20:
        keyboard_weight = 0.1
        mouse_weight = 0.9

    # Cenário B: Uso pesado de atalhos (Copiar/Colar)
    elif len(shortcuts_data) > 0 and len(keyboard_data) < 5:
        keyboard_weight = 0.1
        mouse_weight = 0.6
        shortcut_weight = 0.3

    # 4. SALVAMENTO NO MONGODB
    if keyboard_data:
        keystrokes_collection.insert_many(keyboard_data)
        
    if mouse_data:
        mouse_events_collection.insert_many(mouse_data)
        
    # Atalhos podem ser salvos na coleção de keystrokes com tipo diferenciado
    if shortcuts_data:
        keystrokes_collection.insert_many(shortcuts_data)

    return jsonify({
        "status": "success", 
        "message": "Dados biométricos salvos no MongoDB",
        "dynamic_weights": {
            "keyboard": keyboard_weight,
            "mouse": mouse_weight,
            "shortcut": shortcut_weight
        }
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
    # AQUI ENTRA A IA FUTURA: O sistema puxará os dados do banco e rodará o modelo
    
    # Para testes preliminares, retornamos um score simulado
    fake_score = 0.92
    limiar_seguranca = 0.80

    status_sessao = "legitimo" if fake_score >= limiar_seguranca else "suspeito"

    return jsonify({
        "status": "success",
        "score": fake_score,
        "resultado": status_sessao,
        "mensagem": "Análise comportamental concluída"
    }), 200