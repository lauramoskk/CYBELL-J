"""01_extrair_dados.py

Conecta ao MongoDB (lendo MONGO_URI do .env) e transforma cada coleção
da base `cybell_db` em um pandas DataFrame.

Uso: python 01_extrair_dados.py
"""
from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB", "cybell_db")

if not URI:
    raise RuntimeError("MONGO_URI não está definida. Defina em .env ou como variável de ambiente.")


def get_db(uri: str, db_name: str):
    client = MongoClient(uri)
    return client[db_name]


def collection_to_dataframe(db, collection_name: str, limit: int | None = None) -> pd.DataFrame:
    """Carrega documentos da coleção e retorna um DataFrame plano (json_normalize).

    - Converte `_id` para string para evitar tipos incompatíveis.
    - `limit` pode ser usado para depuração.
    """
    cursor = db[collection_name].find()
    if limit:
        cursor = cursor.limit(limit)
    docs = list(cursor)
    if not docs:
        return pd.DataFrame()

    for d in docs:
        if "_id" in d:
            d["_id"] = str(d["_id"])

    df = pd.json_normalize(docs)
    return df


def main():
    db = get_db(URI, DB_NAME)
    collections = db.list_collection_names()
    if not collections:
        print("Nenhuma coleção encontrada em", DB_NAME)
        return

    print("Coleções encontradas:", collections)

    dataframes = {}
    for coll in collections:
        df = collection_to_dataframe(db, coll)
        print(f"\nColeção: {coll} — linhas: {df.shape[0]} colunas: {df.shape[1]}")
        if not df.empty:
            print(df.head(3).to_string(index=False))
        else:
            print("(vazia)")
        dataframes[coll] = df

    # Cria pasta data/ e salva cada DataFrame como CSV
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    for coll, df in dataframes.items():
        if df.empty:
            print(f"Coleção {coll} vazia — pulando salvar CSV")
            continue
        out_path = out_dir / f"{coll}.csv"
        df.to_csv(out_path, index=False)
        print(f"Salvo: {out_path}")

    # Retorna o dicionário de DataFrames caso alguém importe este módulo.
    return dataframes


if __name__ == "__main__":
    main()
