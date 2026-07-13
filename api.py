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
    Recebe dados comportamentais (teclado e mouse) do Front-end
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

    # Salva no MongoDB se houver dados de teclado
    if keyboard_data:
        keystrokes_collection.insert_many(keyboard_data)

    # Salva no MongoDB se houver dados de mouse
    if mouse_data:
        mouse_events_collection.insert_many(mouse_data)

    return jsonify({"status": "success", "message": "Dados biométricos salvos no MongoDB"}), 200

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
    # AQUI ENTRA A IA FUTURA: O sistema puxará os dados do banco e rodará o modelo (ex: Random Forest)
    
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