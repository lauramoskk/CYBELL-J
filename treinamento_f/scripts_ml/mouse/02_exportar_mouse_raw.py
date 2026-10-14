import os
import csv
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from pymongo import MongoClient
from dotenv import load_dotenv


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

uri = os.getenv("MONGO_URI")
db_name = os.getenv("MONGO_DB", "cybell_db")

if not uri:
    raise RuntimeError(
        "MONGO_URI não encontrada no arquivo .env"
    )


# ============================================================
# PASTA DE SAÍDA
# ============================================================

DATA_DIR = BASE_DIR / "data" / "mouse_snapshot"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

csv_path = DATA_DIR / "mouse_events_raw.csv"
metadata_path = DATA_DIR / "mouse_events_metadata.json"


# ============================================================
# CONEXÃO — SOMENTE LEITURA
# ============================================================

print("=" * 70)
print("EXPORTAÇÃO SEGURA DOS DADOS DE MOUSE/TRACKPAD")
print("=" * 70)

print("\nConectando ao MongoDB...")

client = MongoClient(
    uri,
    serverSelectionTimeoutMS=10000
)

client.admin.command("ping")

print("✅ Conexão estabelecida.")

db = client[db_name]
collection = db["mouse_events"]


# ============================================================
# CONGELA O PONTO FINAL DA EXTRAÇÃO
# ============================================================
#
# Usamos o maior _id existente neste momento.
# Assim, registros novos que entrarem depois não serão incluídos.
#
# _id já possui índice no MongoDB.
#

ultimo_documento = collection.find_one(
    {},
    {"_id": 1},
    sort=[("_id", -1)]
)

if not ultimo_documento:
    raise RuntimeError(
        "A coleção mouse_events está vazia."
    )

cutoff_id = ultimo_documento["_id"]

filtro_snapshot = {
    "_id": {
        "$lte": cutoff_id
    }
}


# ============================================================
# CONTAGEM DO SNAPSHOT
# ============================================================

total_snapshot = collection.count_documents(
    filtro_snapshot
)

print(
    f"\nRegistros congelados neste snapshot: "
    f"{total_snapshot}"
)

print(
    f"Último _id incluído: {cutoff_id}"
)


# ============================================================
# CAMPOS PRESERVADOS
# ============================================================

campos = [
    "_id",
    "user",
    "timestamp",
    "event_type",
    "device_type",
    "x",
    "y",
    "delta_x",
    "delta_y"
]

projection = {
    campo: 1
    for campo in campos
}


# ============================================================
# CONSULTA SOMENTE LEITURA
# ============================================================
#
# A ordenação agora é feita por _id, que possui índice.
# Não ordenamos por timestamp dentro do Mongo.
#

cursor = (
    collection
    .find(
        filtro_snapshot,
        projection
    )
    .sort(
        "_id",
        1
    )
    .batch_size(5000)
)


# ============================================================
# EXPORTAÇÃO
# ============================================================

print("\nExportando dados...")

contador = 0

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:

    writer = csv.DictWriter(
        arquivo,
        fieldnames=campos
    )

    writer.writeheader()

    for documento in cursor:

        linha = {}

        for campo in campos:

            linha[campo] = documento.get(
                campo,
                ""
            )

        linha["_id"] = str(
            linha["_id"]
        )

        writer.writerow(
            linha
        )

        contador += 1

        if contador % 50000 == 0:

            print(
                f"  {contador} registros exportados..."
            )


print(
    f"\n✅ {contador} registros exportados."
)


# ============================================================
# VERIFICAÇÃO
# ============================================================

if contador != total_snapshot:

    print("\n⚠️ ATENÇÃO")

    print(
        "A quantidade exportada não corresponde "
        "à quantidade esperada."
    )

    print(
        f"Esperado: {total_snapshot}"
    )

    print(
        f"Exportado: {contador}"
    )

else:

    print(
        "\n✅ Quantidade exportada confere com o snapshot."
    )


# ============================================================
# SHA-256
# ============================================================

print(
    "\nCalculando assinatura SHA-256..."
)

sha256 = hashlib.sha256()

with open(
    csv_path,
    "rb"
) as arquivo:

    while True:

        bloco = arquivo.read(
            1024 * 1024
        )

        if not bloco:
            break

        sha256.update(
            bloco
        )


hash_arquivo = sha256.hexdigest()

print(
    f"SHA-256: {hash_arquivo}"
)


# ============================================================
# METADADOS
# ============================================================

metadata = {

    "projeto": "CYBELL-J",

    "colecao_origem": "mouse_events",

    "database": db_name,

    "data_extracao_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "cutoff_object_id": str(
        cutoff_id
    ),

    "registros_snapshot": total_snapshot,

    "registros_exportados": contador,

    "arquivo": csv_path.name,

    "sha256": hash_arquivo,

    "campos": campos,

    "observacao": (
        "Snapshot gerado exclusivamente com "
        "operações de leitura. Nenhum registro "
        "do MongoDB foi inserido, alterado ou excluído."
    )
}


with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        metadata,
        arquivo,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)

print("SNAPSHOT CONCLUÍDO")

print("=" * 70)

print(
    f"\nArquivo CSV:\n{csv_path}"
)

print(
    f"\nArquivo de metadados:\n{metadata_path}"
)

print(
    f"\nRegistros exportados: {contador}"
)

print(
    f"\nSHA-256:\n{hash_arquivo}"
)

print(
    "\n✅ Nenhum dado do MongoDB foi alterado."
)

client.close()