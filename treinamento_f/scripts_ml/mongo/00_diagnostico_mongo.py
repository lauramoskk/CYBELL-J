import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

uri = os.getenv("MONGO_URI")
db_name = os.getenv("MONGO_DB", "cybell_db")

if not uri:
    raise RuntimeError(
        "MONGO_URI não encontrada. Verifique se o arquivo .env está na raiz do projeto."
    )

print("Tentando conectar ao MongoDB...")

client = MongoClient(
    uri,
    serverSelectionTimeoutMS=10000
)

# Apenas testa a conexão
client.admin.command("ping")

print("✅ Conexão com MongoDB funcionando.")

db = client[db_name]

# Apenas leitura
colecoes = db.list_collection_names()

print("\n=== COLEÇÕES ENCONTRADAS ===")

for nome in colecoes:
    quantidade = db[nome].count_documents({})
    print(f"{nome}: {quantidade} registros")


# Auditoria específica de mouse
if "mouse_events" in colecoes:

    coll = db["mouse_events"]

    print("\n=== AUDITORIA DE MOUSE_EVENTS ===")

    exemplo = coll.find_one()

    if exemplo:
        print("\nCampos existentes:")

        for campo in sorted(exemplo.keys()):
            print(" -", campo)

    print("\nQuantidade por tipo de evento:")

    resultado = coll.aggregate([
        {
            "$group": {
                "_id": "$event_type",
                "quantidade": {"$sum": 1}
            }
        },
        {
            "$sort": {
                "quantidade": -1
            }
        }
    ])

    for item in resultado:
        print(
            f"{item['_id']}: "
            f"{item['quantidade']}"
        )

    print("\nQuantidade por device_type:")

    resultado = coll.aggregate([
        {
            "$group": {
                "_id": "$device_type",
                "quantidade": {"$sum": 1}
            }
        },
        {
            "$sort": {
                "quantidade": -1
            }
        }
    ])

    for item in resultado:
        print(
            f"{item['_id']}: "
            f"{item['quantidade']}"
        )

    usuarios = coll.distinct("user")

    print("\nQuantidade de usuários encontrados:")
    print(len(usuarios))

    print("\nUsuários:")
    for usuario in usuarios:
        print(" -", usuario)

    primeiro = coll.find_one(
        sort=[("timestamp", 1)]
    )

    ultimo = coll.find_one(
        sort=[("timestamp", -1)]
    )

    if primeiro:
        print("\nPrimeiro registro:")
        print(primeiro.get("timestamp"))

    if ultimo:
        print("\nÚltimo registro:")
        print(ultimo.get("timestamp"))

else:
    print(
        "\n⚠️ A coleção mouse_events não foi encontrada."
    )

print("\n✅ Diagnóstico concluído.")
print("Nenhum dado foi alterado.")