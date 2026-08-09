from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv
from bson import json_util

load_dotenv()

URI = os.getenv("MONGO_URI")
if not URI:
    raise RuntimeError("MONGO_URI não está definida. Defina em .env ou como variável de ambiente.")

client = MongoClient(URI)

db = client["cybell_db"]

print("Coleções disponíveis:")
print(db.list_collection_names())

print("\nQuantidade de documentos:")

for collection in db.list_collection_names():
    print(collection, db[collection].count_documents({}))

print('\nExemplo de documento por coleção:')
for collection in db.list_collection_names():
    doc = db[collection].find_one()
    print(f"--- {collection} ---")
    if doc:
        print(json_util.dumps(doc, indent=2))
    else:
        print("(vazio)")