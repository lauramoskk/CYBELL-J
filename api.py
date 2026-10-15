from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.security import check_password_hash
from mongo import keystrokes_collection, mouse_events_collection
from scripts_ml.inference.mouse_inference import prever_mouse
from scripts_ml.inference.keyboard_inference import prever_teclado
import math
import time

# ============================================
# Blueprint das rotas da API
# ============================================

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

    if not all(
        isinstance(items, list)
        for items in (keyboard_data, mouse_data, shortcuts_data)
    ):
        return jsonify({
            "status": "error",
            "message": "Os eventos devem ser enviados em listas."
        }), 400

    if not keyboard_data and not mouse_data and not shortcuts_data:
        return jsonify({
            "status": "empty_window",
            "message": "Nenhum dado comportamental recebido nesta janela."
        }), 200

    if len(mouse_data) > 400 or len(keyboard_data) > 200:
        return jsonify({
            "status": "security_violation",
            "message": "Volume excessivo de eventos detectado."
        }), 429

    username = current_user.username
    sanitized_keyboard = []
    sanitized_mouse = []
    sanitized_shortcuts = []

    try:
        # Eventos de teclado
        for event in keyboard_data:
            if not isinstance(event, dict):
                continue

            timestamp = int(event.get("timestamp", time.time()))

            sanitized_keyboard.append({
                "user": username,
                "key": str(event.get("key", ""))[:30],
                "event_type": str(
                    event.get("event_type", "keydown")
                )[:30],
                "timestamp": timestamp
            })

        # Eventos de mouse
        for event in mouse_data:
            if not isinstance(event, dict):
                continue

            x_val = float(event.get("x", 0.0))
            y_val = float(event.get("y", 0.0))
            delta_x = float(event.get("delta_x", 0.0))
            delta_y = float(event.get("delta_y", 0.0))
            timestamp = int(event.get("timestamp", time.time()))

            if not all(math.isfinite(value) for value in (
                x_val, y_val, delta_x, delta_y
            )):
                continue

            if not (0 <= x_val <= 4000 and 0 <= y_val <= 3000):
                continue

            sanitized_mouse.append({
                "user": username,
                "event_type": str(event.get("event_type", ""))[:30],
                "device_type": str(
                    event.get("device_type", "mouse")
                )[:20],
                "x": x_val,
                "y": y_val,
                "delta_x": delta_x,
                "delta_y": delta_y,
                "timestamp": timestamp
            })

        # Atalhos de teclado
        for event in shortcuts_data:
            if not isinstance(event, dict):
                continue

            hold_time = float(event.get("hold_time", 0.0))
            timestamp = int(event.get("timestamp", time.time()))

            if not math.isfinite(hold_time) or hold_time < 0:
                continue

            sanitized_shortcuts.append({
                "user": username,
                "key": str(event.get("key", ""))[:30],
                "event_type": "shortcut",
                "hold_time": hold_time,
                "timestamp": timestamp
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
            "message": "Um ou mais eventos contêm valores inválidos."
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
# Funções auxiliares de avaliação
# ============================================

def calcular_margem(probabilidades):
    """Retorna a diferença entre as duas maiores probabilidades."""
    valores = sorted(probabilidades.values(), reverse=True)

    if not valores:
        return 0.0

    maior = valores[0]
    segunda = valores[1] if len(valores) > 1 else 0.0

    return float(maior - segunda)


def normalizar_usuario(usuario):
    """Normaliza o nome para comparação sem diferenciar maiúsculas."""
    return str(usuario).strip().casefold()


def avaliar_confianca(confianca, margem):
    """
    Critério provisório de confiança e margem.
    Os limiares precisam ser validados com dados independentes.
    """
    return (
        math.isfinite(confianca)
        and math.isfinite(margem)
        and confianca >= 0.80
        and margem >= 0.20
    )


# ============================================
# Verificação comportamental pela IA
# ============================================

@api_bp.route("/api/verify", methods=["POST"])
@login_required
def verify_ia():
    """
    Compara previsões de mouse e teclado.

    Os resultados dos modelos não comprovam, por si só, a identidade
    de uma pessoa. Os critérios de decisão são provisórios.
    """
    tamanho_janela = 50
    usuario_logado = str(current_user.username).strip()
    usuario_logado_normalizado = normalizar_usuario(usuario_logado)

    try:
        # 1. Buscar os eventos de mouse
        eventos_mouse = list(
            mouse_events_collection
            .find({"user": current_user.username})
            .sort("timestamp", -1)
            .limit(tamanho_janela)
        )

        # 2. Buscar os eventos de teclado
        eventos_teclado = list(
            keystrokes_collection
            .find({"user": current_user.username})
            .sort("timestamp", -1)
            .limit(tamanho_janela)
        )

        # 3. Verificar se há dados suficientes
        if (
            len(eventos_mouse) < tamanho_janela
            or len(eventos_teclado) < tamanho_janela
        ):
            return jsonify({
                "status": "insufficient_data",
                "resultado": "indisponivel",
                "eventos_mouse_analisados": len(eventos_mouse),
                "eventos_teclado_analisados": len(eventos_teclado),
                "mensagem": (
                    "Dados insuficientes para a verificação conjunta. "
                    f"São necessários {tamanho_janela} eventos de mouse "
                    f"e {tamanho_janela} eventos de teclado."
                )
            }), 400

        # Remover o identificador interno do MongoDB
        for evento in eventos_mouse:
            evento.pop("_id", None)

        for evento in eventos_teclado:
            evento.pop("_id", None)

        # 4. Executar os dois modelos
        resultado_mouse = prever_mouse(eventos_mouse)
        resultado_teclado = prever_teclado(eventos_teclado)

        usuario_mouse = str(
            resultado_mouse["usuario_previsto"]
        ).strip()

        usuario_teclado = str(
            resultado_teclado["usuario_previsto"]
        ).strip()

        confianca_mouse = float(resultado_mouse["confianca"])
        confianca_teclado = float(resultado_teclado["confianca"])

        probabilidades_mouse = {
            str(usuario): float(probabilidade)
            for usuario, probabilidade
            in resultado_mouse["probabilidades"].items()
        }

        probabilidades_teclado = {
            str(usuario): float(probabilidade)
            for usuario, probabilidade
            in resultado_teclado["probabilidades"].items()
        }

        # 5. Validar as probabilidades
        for nome_modelo, confianca, probabilidades in (
            ("mouse", confianca_mouse, probabilidades_mouse),
            ("teclado", confianca_teclado, probabilidades_teclado)
        ):
            valores = list(probabilidades.values())

            if (
                not math.isfinite(confianca)
                or not 0.0 <= confianca <= 1.0
                or not valores
                or any(
                    not math.isfinite(valor)
                    or not 0.0 <= valor <= 1.0
                    for valor in valores
                )
            ):
                raise ValueError(
                    f"Probabilidades inválidas retornadas pelo modelo "
                    f"de {nome_modelo}."
                )

        # 6. Calcular as margens
        margem_mouse = calcular_margem(probabilidades_mouse)
        margem_teclado = calcular_margem(probabilidades_teclado)

        mouse_normalizado = normalizar_usuario(usuario_mouse)
        teclado_normalizado = normalizar_usuario(usuario_teclado)

        mouse_forte = avaliar_confianca(
            confianca_mouse,
            margem_mouse
        )

        teclado_forte = avaliar_confianca(
            confianca_teclado,
            margem_teclado
        )

        # 7. Decidir o resultado

        # Confirmação provisória: ambos preveem o usuário autenticado
        # e atendem aos critérios mínimos de confiança e margem.
        if (
            mouse_normalizado == usuario_logado_normalizado
            and teclado_normalizado == usuario_logado_normalizado
        ):
            if mouse_forte and teclado_forte:
                resultado = "confirmado"
                mensagem = (
                    "Os modelos de mouse e teclado previram o usuário "
                    "autenticado e atenderam aos critérios provisórios "
                    "de confiança e margem."
                )
            else:
                resultado = "inconclusivo"
                mensagem = (
                    "Os modelos de mouse e teclado previram o usuário "
                    "autenticado, mas um ou ambos não atingiram os "
                    "critérios provisórios de confiança e margem."
                )

        # Suspeita: ambos preveem o mesmo usuário diferente do autenticado
        # e ambos atendem aos critérios mínimos.
        elif (
            mouse_normalizado == teclado_normalizado
            and mouse_normalizado != usuario_logado_normalizado
            and mouse_forte
            and teclado_forte
        ):
            resultado = "suspeito"
            mensagem = (
                "Os modelos de mouse e teclado previram o mesmo usuário "
                "diferente da conta autenticada. Ambos atenderam aos "
                "critérios provisórios de confiança e margem. Isso é um "
                "alerta para avaliação, não uma prova de identidade."
            )

        # Divergência ou confiança insuficiente
        else:
            resultado = "inconclusivo"
            mensagem = (
                "Os modelos não apresentaram concordância suficiente "
                "para sinalizar suspeita segundo os critérios "
                "provisórios. É necessária avaliação adicional."
            )

        # 8. Retornar os resultados
        return jsonify({
            "status": "success",
            "resultado": resultado,
            "usuario_autenticado": usuario_logado,
            "usuario_previsto_mouse": usuario_mouse,
            "score_mouse": confianca_mouse,
            "margem_mouse": margem_mouse,
            "probabilidades_mouse": probabilidades_mouse,
            "usuario_previsto_teclado": usuario_teclado,
            "score_teclado": confianca_teclado,
            "margem_teclado": margem_teclado,
            "probabilidades_teclado": probabilidades_teclado,
            "eventos_mouse_analisados": len(eventos_mouse),
            "eventos_teclado_analisados": len(eventos_teclado),
            "mensagem": mensagem
        }), 200

    except ValueError as error:
        current_app.logger.warning(
            "Dados inválidos na verificação comportamental: %s",
            error
        )
        return jsonify({
            "status": "error",
            "message": (
                "Não foi possível processar os dados "
                "da verificação comportamental."
            )
        }), 400

    except Exception:
        current_app.logger.exception(
            "Erro durante a verificação de mouse e teclado."
        )
        return jsonify({
            "status": "error",
            "message": "Erro durante a análise comportamental."
        }), 500