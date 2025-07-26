import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "Sparkly-ai-db"

def create_collection():
    if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

def add_to_vector_store(chunks, embeddings):
    points = [PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i]}) for i in range(len(chunks))]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_similar(query_embedding, top_k=3):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )
    return [res.payload['text'] for res in results]
