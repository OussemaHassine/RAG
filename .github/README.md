# 📄 DocuRAG — Advanced RAG Pipeline for Document Q&A
 
A production-grade Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and have an intelligent conversation about its contents. Built with advanced retrieval techniques, hybrid search, and conversational memory.
 
---
 
## 🚀 Live Demo
 
> [Deployed on Hugging Face Spaces ](https://oussemahassine-rag-hf.hf.space/)
 
---
 
## 🧠 What Makes This GOOD
 
Although it might seem like an overkill for a personal project, I wanted to implement advanced and sophisticated approaches to learn the most!

| Component | Technique |
|---|---|
| **Chunking** | Semantic chunking (sentence-transformers) + Recursive 512-Token |
| **Embeddings** | OpenAI `text-embedding-3-small` (dense, 1536d) |
| **Sparse Vectors** | BM25 via FastEmbed (`Qdrant/bm25`) |
| **Retrieval** | Hybrid search (dense + sparse) with Reciprocal Rank Fusion (RRF) |
| **Reranking** | Cohere Rerank v3.5 |
| **Generation** | GPT-4o-mini with structured prompt engineering |
| **Memory** | Sliding window + LLM-based summarization of older turns |
| **Vector Store** | Qdrant Cloud (free tier, persistent) |
| **UI** | Streamlit with streaming responses |
 
---
 
 ## 🧪 Evaluation
 
The project includes a RAGAS evaluation pipeline (`evaluation/evaluate.py`) that measures:
- **Faithfulness** — are answers grounded in the retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are the retrieved chunks actually relevant?
- **Context Recall** — are all relevant chunks being retrieved?

[Based on a single Erasmus Italian PDF document that has 23 pages](https://www.uniurb.it/it/cdocs/INT/10047-INT-04122025173718-int_bando.pdf), the scores were: 
- faithfulness: 0.8807
- answer_relevancy: 0.7479
- llm_context_precision_without_reference: 0.8843

Results saved to evaluation_results.csv
 
---

## 👤 Author
 
Built by **Oussama Hassine** as a portfolio project while transitioning into AI Engineering.
- LinkedIn: [Oussama Hassine](https://linkedin.com/in/OussemaHassine)
