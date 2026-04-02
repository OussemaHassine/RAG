from clients import qdrant_client, client, sparse_model, cohere_client
from qdrant_client.models import Prefetch, Fusion, SparseVector, models

def embed_query_dense(query: str) -> list[float]:
    """Embed a query string using OpenAI embeddings."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding

def embed_query_sparse(query: str) -> SparseVector:
    """Embed a query string using the sparse embedding model."""
    result = list(sparse_model.embed([query]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist()
    )

from qdrant_client import models

def rrf_search(collection_name: str, query: str, top_k: int = 5) -> list[str]:
    query_dense = embed_query_dense(query)
    query_sparse = embed_query_sparse(query)
    
    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_sparse,
                using="sparse",
                limit=top_k * 2
            ),
            models.Prefetch(
                query=query_dense,
                using="dense",
                limit=top_k * 2
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    )
    return [result.payload["text"] for result in results.points]

def rerank_results(query: str, texts: list[str], top_k: int = 5) -> list[str]:
    """Rerank retrieved chunks using a cross-encoder model."""
    ranker= cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=texts,
        top_n=top_k
    )
    return [texts[result.index] for result in ranker.results]  # Return top_k results as-is for now

def retrieve(collection_name: str, query: str, top_k: int = 5) -> list[str]:
    """Retrieve relevant chunks for a query."""
    initial_results = rrf_search(collection_name, query, top_k*2)  # Get more results for reranking
    reranked_results = rerank_results(query, initial_results, top_k)
    return reranked_results