from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime
from collections import defaultdict

load_dotenv()

uri = os.getenv("MONGO_URI")
db_name = os.getenv("MONGO_DB", "cybell_db")

if not uri:
    raise RuntimeError("MONGO_URI não encontrada no arquivo .env")

client = MongoClient(uri)

# Testa a conexão
client.admin.command("ping")

db = client[db_name]
coll = db["mouse_events"]

print("=" * 70)
print("AUDITORIA DETALHADA DOS DADOS DE MOUSE/TRACKPAD")
print("=" * 70)

# --------------------------------------------------
# 1. TOTAL POR USUÁRIA
# --------------------------------------------------

print("\n1. TOTAL DE EVENTOS POR USUÁRIA\n")

pipeline = [
    {
        "$group": {
            "_id": "$user",
            "total": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "total": -1
        }
    }
]

for item in coll.aggregate(pipeline):
    print(f"{item['_id']:<25} {item['total']:>10} eventos")


# --------------------------------------------------
# 2. MOUSE X TRACKPAD POR USUÁRIA
# --------------------------------------------------

print("\n" + "=" * 70)
print("2. DEVICE_TYPE POR USUÁRIA")
print("=" * 70)

pipeline = [
    {
        "$group": {
            "_id": {
                "user": "$user",
                "device": "$device_type"
            },
            "total": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "_id.user": 1,
            "_id.device": 1
        }
    }
]

dados_usuario = defaultdict(dict)

for item in coll.aggregate(pipeline):

    usuario = item["_id"]["user"]
    device = item["_id"]["device"]
    total = item["total"]

    dados_usuario[usuario][device] = total


for usuario in sorted(dados_usuario):

    mouse = dados_usuario[usuario].get("mouse", 0)
    trackpad = dados_usuario[usuario].get("trackpad", 0)

    total = mouse + trackpad

    if total > 0:
        perc_mouse = mouse / total * 100
        perc_trackpad = trackpad / total * 100
    else:
        perc_mouse = 0
        perc_trackpad = 0

    print(f"\nUsuária: {usuario}")

    print(
        f"  mouse:    {mouse:>8} "
        f"({perc_mouse:6.2f}%)"
    )

    print(
        f"  trackpad: {trackpad:>8} "
        f"({perc_trackpad:6.2f}%)"
    )

    print(
        f"  TOTAL:    {total:>8}"
    )


# --------------------------------------------------
# 3. EVENTOS POR TIPO E USUÁRIA
# --------------------------------------------------

print("\n" + "=" * 70)
print("3. TIPOS DE EVENTO POR USUÁRIA")
print("=" * 70)

pipeline = [
    {
        "$group": {
            "_id": {
                "user": "$user",
                "event": "$event_type"
            },
            "total": {"$sum": 1}
        }
    },
    {
        "$sort": {
            "_id.user": 1,
            "total": -1
        }
    }
]

usuario_atual = None

for item in coll.aggregate(pipeline):

    usuario = item["_id"]["user"]
    evento = item["_id"]["event"]
    total = item["total"]

    if usuario != usuario_atual:

        print(f"\nUsuária: {usuario}")

        usuario_atual = usuario

    print(
        f"  {evento:<20} {total:>8}"
    )


# --------------------------------------------------
# 4. EVENTOS POR DIA E USUÁRIA
# --------------------------------------------------

print("\n" + "=" * 70)
print("4. EVENTOS POR DIA E USUÁRIA")
print("=" * 70)

pipeline = [

    {
        "$project": {

            "user": 1,

            "data": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": {
                        "$toDate": "$timestamp"
                    },
                    "timezone": "America/Sao_Paulo"
                }
            }
        }
    },

    {
        "$group": {

            "_id": {
                "user": "$user",
                "data": "$data"
            },

            "total": {
                "$sum": 1
            }
        }
    },

    {
        "$sort": {
            "_id.data": 1,
            "_id.user": 1
        }
    }

]

data_atual = None

for item in coll.aggregate(pipeline):

    data = item["_id"]["data"]
    usuario = item["_id"]["user"]
    total = item["total"]

    if data != data_atual:

        print(f"\nDATA: {data}")

        data_atual = data

    print(
        f"  {usuario:<25} {total:>8}"
    )


# --------------------------------------------------
# 5. MOUSE/TRACKPAD POR DIA E USUÁRIA
# --------------------------------------------------

print("\n" + "=" * 70)
print("5. MOUSE X TRACKPAD POR DIA E USUÁRIA")
print("=" * 70)

pipeline = [

    {
        "$project": {

            "user": 1,
            "device_type": 1,

            "data": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": {
                        "$toDate": "$timestamp"
                    },
                    "timezone": "America/Sao_Paulo"
                }
            }
        }
    },

    {
        "$group": {

            "_id": {
                "user": "$user",
                "data": "$data",
                "device": "$device_type"
            },

            "total": {
                "$sum": 1
            }
        }
    },

    {
        "$sort": {
            "_id.data": 1,
            "_id.user": 1,
            "_id.device": 1
        }
    }

]

for item in coll.aggregate(pipeline):

    print(
        f"{item['_id']['data']} | "
        f"{item['_id']['user']:<20} | "
        f"{str(item['_id']['device']):<10} | "
        f"{item['total']:>8}"
    )


# --------------------------------------------------
# 6. PRIMEIRO E ÚLTIMO EVENTO POR USUÁRIA
# --------------------------------------------------

print("\n" + "=" * 70)
print("6. PERÍODO DE COLETA POR USUÁRIA")
print("=" * 70)

usuarios = coll.distinct("user")

for usuario in sorted(usuarios):

    primeiro = coll.find_one(
        {"user": usuario},
        sort=[("timestamp", 1)]
    )

    ultimo = coll.find_one(
        {"user": usuario},
        sort=[("timestamp", -1)]
    )

    if primeiro and ultimo:

        dt_inicio = datetime.fromtimestamp(
            primeiro["timestamp"] / 1000
        )

        dt_fim = datetime.fromtimestamp(
            ultimo["timestamp"] / 1000
        )

        print(f"\n{usuario}")

        print(
            "  Primeiro evento:",
            dt_inicio.strftime("%d/%m/%Y %H:%M:%S")
        )

        print(
            "  Último evento:  ",
            dt_fim.strftime("%d/%m/%Y %H:%M:%S")
        )


print("\n" + "=" * 70)
print("AUDITORIA CONCLUÍDA")
print("Nenhum dado foi inserido, alterado ou excluído.")
print("=" * 70)

client.close()