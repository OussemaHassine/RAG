import os
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference
from dotenv import load_dotenv
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from clients import client
from ingestion import get_chunks
from retrieval import retrieve
from prompt import get_answer_non_streaming

load_dotenv()

# ---- LLMs and Embeddings ----
generator_llm = llm_factory("gpt-4o-mini", client=client)
embedding_model = RagasOpenAIEmbeddings(model="text-embedding-3-small", client=client)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# ---- Convert chunks to LangChain Documents ----
def get_documents(path: str, method: str = "recursive") -> list[Document]:
    chunks = get_chunks(path, method=method)
    return [
        Document(
            page_content=chunk.text,
            metadata={"filename": chunk.source_filename}
        )
        for chunk in chunks
    ]

# ---- Generate testset (with caching) ----
def generate_testset(path: str, num_samples: int = 20) -> pd.DataFrame:
    cache_path = "testset_cache.csv"
    if os.path.exists(cache_path):
        print("Loading cached testset...")
        return pd.read_csv(cache_path)

    print("Generating testset (this takes a while)...")
    generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model)
    docs = get_documents(path)
    testset = generator.generate_with_langchain_docs(docs, testset_size=num_samples)
    df = testset.to_pandas()
    df.to_csv(cache_path, index=False)
    print(f"Testset saved to {cache_path}")
    return df

# ---- Run RAG + Evaluate ----
def evaluate_rag(df: pd.DataFrame, collection_name: str):
    questions = df["user_input"].tolist()
    truth = df["reference"].tolist()

    generated_answers = []
    retrieved_chunks_list = []

    print(f"Running RAG on {len(questions)} questions...")
    for i, question in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {question[:60]}...")
        retrieved_chunks = retrieve(collection_name, question, top_k=5)
        retrieved_chunks_list.append(retrieved_chunks)
        answer = get_answer_non_streaming(question, retrieved_chunks)
        generated_answers.append(answer)
        time.sleep(7)  # avoid Cohere rate limit

    # Build RAGAS EvaluationDataset
    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=r
        )
        for q, a, c, r in zip(questions, generated_answers, retrieved_chunks_list, truth)
    ]
    dataset = EvaluationDataset(samples=samples)

    print("Evaluating with RAGAS...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print(results)
    df_results = results.to_pandas()
    df_results.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")
    return results


if __name__ == "__main__":
    df = generate_testset("bando.pdf", num_samples=20)
    print(df[["user_input", "reference"]].head())
    results = evaluate_rag(df, collection_name="bando_recursive")