from ingestion import get_chunks
from enrichment import add_summary
from storage import store_chunks
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
load_dotenv()
from clients import qdrant_client
from retrieval import retrieve
from prompt import get_answer

def verify_file_existance_in_qdrant(collection_name: str, source_filename: str, chunk_method: str) -> bool:
    """Check if a file has already been processed and stored in Qdrant and by which chunking method."""
    try:
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_filename",
                        match=MatchValue(value=source_filename),
                    
                    ),
                    FieldCondition(
                        key="chunk_method",
                        match=MatchValue(value=chunk_method),
                    )
                ]
            ),
            limit=1
        )
        return len(points) > 0
    except Exception as e:
        print(f"Error checking file existence in Qdrant: {e}")
        return False

def process_document(path: str, collection_name: str, method: str = "semantic"):
    """Full pipeline to process a document and store it in Qdrant."""
    if verify_file_existance_in_qdrant(collection_name, os.path.basename(path), method):
        print(f"Document '{os.path.basename(path)}' with chunking method '{method}' already exists in collection '{collection_name}'. Skipping processing.")
    else:
        print(f"Processing document: {path}")
        chunks = get_chunks(path, method)
        # print("Adding summaries to chunks...")
        # enriched_chunks = add_summary(chunks)
        print("Storing chunks in Qdrant...")
        store_chunks(chunks, collection_name)
        print("Document processing complete.")

""" process_document("bando.pdf", "recursive_collection", method="recursive")
query = "When is the deadline for submitting the application?"
retrieved_chunks = retrieve("recursive_collection", query, top_k=5)
answer = get_answer(query, retrieved_chunks) """