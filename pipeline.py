from ingestion import get_chunks
from enrichment import add_summary
from storage import store_chunks
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import os
load_dotenv()
from clients import qdrant_client
from retrieval import retrieve
from prompt import get_answer

COLLECTION_NAME = "rag_documents"

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

def process_document(path: str, method: str = "semantic"):
    """Full pipeline to process a document and store it in Qdrant."""
    filename = os.path.basename(path)
    if verify_file_existance_in_qdrant(COLLECTION_NAME, filename, method):
        print(f"Document '{filename}' with chunking method '{method}' already exists. Skipping processing.")
    else:
        print(f"Processing document: {path}")
        chunks = get_chunks(path, method)
        print("Storing chunks in Qdrant...")
        store_chunks(chunks, COLLECTION_NAME)
        print("Document processing complete.")