import fitz  # PyMuPDF
import re
from semantic_chunking import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import Chunk
import os

def extract_text_by_page(pdf_path: str) -> list[str]:
    """Extract text per page, stripping repeated headers/footers heuristically."""
    if hasattr(pdf_path, "read"):  # file-like object from Streamlit uploader
        file_bytes = pdf_path.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    return pages

def clean_page_text(text: str) -> str:
    """
    Clean raw PDF text:
    - Rejoin hyphenated line-breaks (e.g. 'documenta-\ntion' → 'documentation')
    - Collapse mid-sentence line breaks (single \n inside a sentence)
    - Normalize whitespace
    """
    # Rejoin words broken across lines with a hyphen
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Single newline mid-sentence → space (keep double newlines as paragraph breaks)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse extra whitespace
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def get_full_text(pages: list[str]) -> str:
    """Combine cleaned pages into a single text blob."""
    full_text = " ".join(pages)
    return full_text

def get_semantic_chunks(text: list[str], source_filename: str) -> list[Chunk]:
    chunker = SemanticChunker(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        similarity_threshold=0.65,
        max_chunk_size=500,
    )
    chunks = chunker.semantic_chunk(text)
    final_chunks = []
    for i, chunk in enumerate(chunks):
        final_chunks.append(Chunk(
            text=chunk,
            chunk_index=i,
            source_filename=source_filename,
            chunk_method="semantic",
            char_count=len(chunk)
        ))
    final_chunks = [chunk for chunk in final_chunks if chunk.char_count > 50]
    print(f"Total chunks created: {len(final_chunks)}")

    return final_chunks

def get_recursive_chunks(text: str, source_filename: str) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,     # tokens or characters depending config
        chunk_overlap=70
    )
    chunks = splitter.split_text(text)
    print(f"Total chunks created: {len(chunks)}")
    final_chunks = []
    for i, chunk in enumerate(chunks):
        final_chunks.append(Chunk(
            text=chunk,
            chunk_index=i,
            source_filename=source_filename,
            chunk_method="recursive",
            char_count=len(chunk)
        ))
    return final_chunks

def get_chunks(path: str, method: str = "semantic") -> list[Chunk]:
    source_filename = os.path.basename(path)
    pages = extract_text_by_page(path)
    cleaned_pages = [clean_page_text(page) for page in pages]
    text=get_full_text(cleaned_pages)
    if method == "semantic":
        return get_semantic_chunks(text, source_filename)
    elif method == "recursive":
        return get_recursive_chunks(text, source_filename)
    else:
        raise ValueError(f"Unknown chunking method: {method}")
