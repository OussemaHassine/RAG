from clients import client

max_messages = 6  # Max messages to keep in memory

def summarize(messages: list[dict], existing_summary) -> str:
    """Summarize a list of messages into a single string."""
    conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    prompt= f"Existing summary: {existing_summary}\n\nNew conversation:\n{conversation}\n\nUpdate the summary to include the new conversation while keeping it concise."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def update_memory(messages:list, summary: str) -> tuple[list, str]:
    """Update the memory summary with new messages."""
    if len(messages) > max_messages:
        old_messages = messages[:-max_messages//2]
        summary = summarize(old_messages, summary)
        messages = messages[-max_messages//2:]
    return messages, summary

def prompt_with_memory(query: str, chunks: list[str], summary: str, messages: list) -> list[dict]:
    relevant_chunks = "\n\n".join(chunks)
    system_content = "You are a precise legal document assistant. Answer strictly based on the provided context. If the answer is not in the context, say: 'This information is not found in the provided document.'"
    if summary:
        system_content += f"\n\nSummary of earlier conversation:\n{summary}"
    return [
        {"role": "system", "content": system_content},
        *messages,
        {"role": "user", "content": f"Context:\n{relevant_chunks}\n\nQuestion: {query}"}
    ]