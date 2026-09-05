from mongo import mongo_db


def contar_por_usuario(collection, nome_colecao):
    pipeline = [
        {"$group": {"_id": "$user", "total": {"$sum": 1}}},
        {"$sort": {"total": -1}},
    ]

    resultado = list(collection.aggregate(pipeline))

    print(f"\n=== {nome_colecao} ===")
    for item in resultado:
        print(f"Usuario: {item['_id']} | Ocorrências: {item['total']}")

    return resultado


def resumo_por_usuario():
    keystrokes = mongo_db["keystrokes"]
    mouse_events = mongo_db["mouse_events"]

    resultado_teclas = contar_por_usuario(keystrokes, "keystrokes")
    resultado_mouse = contar_por_usuario(mouse_events, "mouse_events")

    usuarios = {}

    for item in resultado_teclas:
        usuarios[item["_id"]] = {"usuario": item["_id"], "keystrokes": item["total"], "mouse_events": 0}

    for item in resultado_mouse:
        if item["_id"] not in usuarios:
            usuarios[item["_id"]] = {"usuario": item["_id"], "keystrokes": 0, "mouse_events": 0}
        usuarios[item["_id"]]["mouse_events"] = item["total"]

    print("\n=== Resumo por usuário ===")
    for usuario in usuarios.values():
        print(usuario)


if __name__ == "__main__":
    resumo_por_usuario()
