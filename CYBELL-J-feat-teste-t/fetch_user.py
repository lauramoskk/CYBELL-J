import sys
from pprint import pprint

try:
    from bson import ObjectId
except Exception:
    ObjectId = None

from mongo import mongo_db


def try_objectid(id_str):
    if not ObjectId:
        return None
    try:
        return ObjectId(id_str)
    except Exception:
        return None


def find_user_by_id(user_id):
    results = []
    oid = try_objectid(user_id)
    # Search all collections for a document with _id equal to user_id (string) or ObjectId
    for col_name in mongo_db.list_collection_names():
        col = mongo_db[col_name]
        # try string id first
        doc = col.find_one({"_id": user_id})
        if doc:
            results.append((col_name, doc))
            continue
        # then try ObjectId if available
        if oid:
            doc = col.find_one({"_id": oid})
            if doc:
                results.append((col_name, doc))
    return results


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else "yass"
    found = find_user_by_id(user_id)
    if not found:
        print(f"Não foi encontrado documento com _id={user_id}")
        return
    for col, doc in found:
        print(f"Encontrado em coleção: {col}")
        pprint(doc)


if __name__ == "__main__":
    main()
