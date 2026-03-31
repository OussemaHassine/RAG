from dataclasses import dataclass
from fastembed import SparseTextEmbedding

@dataclass
class Chunk:
    text: str
    chunk_index: int
    source_filename: str
    chunk_method: str
    char_count: int
    summary: str = ""  # Optional summary field for future use
    embedding: list[float] = None  # To be filled after embedding
    sparse_embedding: dict = None  # To be filled after sparse embedding

sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")