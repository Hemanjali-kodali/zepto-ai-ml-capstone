import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = "support_assistant/documents"
CHROMA_DIR = "support_assistant/chroma_db"


model = SentenceTransformer("all-MiniLM-L6-v2")


client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="zepto_policies"
)


documents = []
ids = []
metadatas = []

for filename in sorted(os.listdir(DOCUMENTS_DIR)):

    if filename.endswith(".txt"):

        file_path = os.path.join(DOCUMENTS_DIR, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        documents.append(text)
        ids.append(filename)

        metadatas.append({
            "source": filename
        })


embeddings = model.encode(documents).tolist()


collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("===== CHROMADB INGESTION =====")
print("Documents loaded:", len(documents))
print("Embeddings created:", len(embeddings))
print("Collection:", collection.name)
print("Documents in collection:", collection.count())