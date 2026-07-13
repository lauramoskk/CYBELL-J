from pymongo import MongoClient

# ==========================================
# CONFIGURAÇÃO DO MONGODB (INTEGRANTE 2)
# ==========================================
# Conecta ao MongoDB local (certifique-se de ter o MongoDB instalado e rodando)
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["cybell_db"]
keystrokes_collection = mongo_db["keystrokes"]
mouse_events_collection = mongo_db["mouse_events"]