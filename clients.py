from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from openai import OpenAI
load_dotenv()
from fastembed import SparseTextEmbedding
import cohere

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))  
qdrant_client = QdrantClient(
    url=os.getenv("CLUSTER_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=20
)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")