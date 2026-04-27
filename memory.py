import chromadb
from uuid import uuid4

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(name="chat_memory")

def save_message(user_id, text):
    collection.add(
        ids=[str(uuid4())],
        documents=[text],
        metadatas=[{"user_id": user_id}]
    )

def get_memory(user_id, query, n_results=2):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"user_id": user_id}
    )

    docs = results.get("documents", [])
    if docs and docs[0]:
        return " ".join(docs[0])

    return ""