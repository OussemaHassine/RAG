from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
load_dotenv()
from models import Chunk
import uuid
from qdrant_client.models import VectorParams, Distance, SparseVectorParams
from qdrant_client.models import PointStruct, SparseVector
from clients import client, qdrant_client, sparse_model


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Add embeddings to a list of chunks."""
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk.text
        )
        chunk.embedding = response.data[0].embedding
    return chunks
def sparse_embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Add sparse embeddings to a list of chunks."""
    for chunk in chunks:
        result = list(sparse_model.embed([chunk.text]))[0]
        chunk.sparse_embedding = {
            "indices": result.indices.tolist(),
            "values": result.values.tolist()
        }
    return chunks

def create_collection(collection_name: str):
    """Create a Qdrant collection if it doesn't exist."""
    existing_collections = qdrant_client.get_collections().collections
    if collection_name not in [col.name for col in existing_collections]:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=1536,  # Size of the embedding vector
                    distance=Distance.COSINE  # Distance metric for similarity search
                )},
                sparse_vectors_config={
                    "sparse": SparseVectorParams()     
                }
        )
        print (f"Collection '{collection_name}' created.")
    else:
        print (f"Collection '{collection_name}' already exists.")

    #we're going to filter by file_name and chunk method to avoid duplicate processing, so we need to create payload indexes for those fields
    qdrant_client.create_payload_index(
    collection_name=collection_name,
    field_name="source_filename",
    field_schema="keyword"
                        )
    qdrant_client.create_payload_index(
    collection_name=collection_name,
    field_name="chunk_method",
    field_schema="keyword"
        )
    
def upsert_chunks(chunks: list[Chunk], collection_name: str):
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        points = []
        for chunk in batch:
            point = PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source_filename}_{chunk.chunk_index}")),
                vector={
                    "dense": chunk.embedding,
                    "sparse": SparseVector(
                        indices=chunk.sparse_embedding["indices"],
                        values=chunk.sparse_embedding["values"]
                    )
                },
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
        print(f"Upserted batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
    print(f"Stored {len(chunks)} chunks in collection '{collection_name}'.")

def store_chunks(chunks: list[Chunk], collection_name: str):
    """Embed and store chunks in Qdrant."""
    create_collection(collection_name)
    print("Dense Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)
    print("Sparse Embedding chunks...")
    sparse_embedded_chunks = sparse_embed_chunks(embedded_chunks)
    print("Upserting chunks to Qdrant...")
    upsert_chunks(sparse_embedded_chunks, collection_name)