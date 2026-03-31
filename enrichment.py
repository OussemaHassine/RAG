from models import Chunk
import os
from openai import OpenAI
import time
from dotenv import load_dotenv
from clients import client

load_dotenv()
def add_summary(chunks: list[Chunk]) -> list[Chunk]:
    """Add a summary to an existing chunk."""
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a document assistant."},
                {"role": "user", "content": f"Summarize this chunk: {chunk.text}"}
            ]
        )
        summary = response.choices[0].message.content
        chunk.summary = summary
        time.sleep(0.5)  # Sleep to respect rate limits
    return chunks