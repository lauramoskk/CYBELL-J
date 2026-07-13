from pymongo import MongoClient

# ==========================================
# CONFIGURAÇÃO DO MONGODB NA NUVEM (ATLAS)
# ==========================================
# Cole a sua URI do MongoDB Atlas aqui (já com a senha real no lugar de <password>):
URI_NUVEM = "mongodb+srv://laurasoliveira2018_db_user:o2vxITy0n9PLY37Q@cybell-j.rtnzeaw.mongodb.net/?appName=Cybell-J"

# Conecta ao MongoDB Atlas
mongo_client = MongoClient(URI_NUVEM)

# Mantém o mesmo nome do banco e das coleções
mongo_db = mongo_client["cybell_db"]
keystrokes_collection = mongo_db["keystrokes"]
mouse_events_collection = mongo_db["mouse_events"]