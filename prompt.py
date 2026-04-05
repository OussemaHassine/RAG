from clients import client

def generate_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """Generate a prompt for the LLM using the retrieved chunks."""
    context = "\n\n".join(retrieved_chunks)
    messages = [
    {
            "role": "system",
            "content": """You are a precise document assistant. Your job is to answer questions strictly based on the provided context extracted from documents.

            Rules:
            - Only use information present in the context
            - If the answer is not in the context, say explicitly: "This information is not found in the provided document."
            - Be concise but complete
            - When referencing specific clauses or articles, mention them explicitly
            - Never make up information"""
                },
                {
                    "role": "user",
                    "content": f"""Context:
            {context}

            Question: {query}

            Answer based strictly on the context above:"""
                }
            ]
    return messages

def generate_response_stream(messages: list[dict]) -> str:
    """Generate a response from the LLM given a list of messages."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        stream=True
    )
    for chunk in response:
        delta= chunk.choices[0].delta.content
        if delta:
            yield delta
def generate_response(messages: list[dict]) -> str: 
    """Generate a response from the LLM given a list of messages."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content

def get_answer(query: str, retrieved_chunks: list[str]) -> str:
    """Get an answer for a query using the retrieved chunks."""
    prompt = generate_prompt(query, retrieved_chunks)
    response = generate_response_stream(prompt)
    print(f"Query: {query}\nAnswer: {response}\n")
    print ("Context used:")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"  {i}. {chunk}")
    return response

def get_answer_non_streaming(query: str, retrieved_chunks: list[str]) -> str:
    """Get an answer for a query using the retrieved chunks."""
    prompt = generate_prompt(query, retrieved_chunks)
    response = generate_response(prompt)
    print(f"Query: {query}\nAnswer: {response}\n")
    print ("Context used:")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"  {i}. {chunk}")
    return response