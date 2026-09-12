import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Pega a URI da variável de ambiente com segurança
URI_NUVEM = os.getenv("MONGO_URI")

if not URI_NUVEM:
    raise ValueError("A variável de ambiente MONGO_URI não foi encontrada. Verifique o arquivo .env")

# Conecta ao MongoDB Atlas
mongo_client = MongoClient(URI_NUVEM)

# Mantém o mesmo nome do banco e das coleções
mongo_db = mongo_client["cybell_db"]
keystrokes_collection = mongo_db["keystrokes"]
mouse_events_collection = mongo_db["mouse_events"]