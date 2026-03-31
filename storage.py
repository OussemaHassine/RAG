from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
load_dotenv()
from models import Chunk
import uuid
from openai import OpenAI
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from clients import client, qdrant_client

def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Add embeddings to a list of chunks."""
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk.text
        )
        chunk.embedding = response.data[0].embedding
    return chunks
def create_collection(collection_name: str):
    """Create a Qdrant collection if it doesn't exist."""
    existing_collections = qdrant_client.get_collections().collections
    if collection_name not in [col.name for col in existing_collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # Size of the embedding vector
                distance=Distance.COSINE  # Distance metric for similarity search
            )
        )
        print (f"Collection '{collection_name}' created.")
    else:
        print (f"Collection '{collection_name}' already exists.")
    
def upsert_chunks(chunks: list[Chunk], collection_name: str):
    """Store chunks with embeddings in Qdrant."""
    points = []
    for chunk in chunks:
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source_filename}_{chunk.chunk_index}")),
            vector=chunk.embedding,
            payload={
                "text": chunk.text,
                "source_filename": chunk.source_filename,
                "chunk_method": chunk.chunk_method,
                "char_count": chunk.char_count,
                "summary": chunk.summary
            }
            )
        points.append(point)
    qdrant_client.upsert(collection_name=collection_name, points=points)

def store_chunks(chunks: list[Chunk], collection_name: str):
    """Embed and store chunks in Qdrant."""
    create_collection(collection_name)
    embedded_chunks = embed_chunks(chunks)
    upsert_chunks(embedded_chunks, collection_name)