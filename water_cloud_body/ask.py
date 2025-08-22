import os
from dotenv import load_dotenv
from openai import OpenAI
from study import VectorStore
from datetime import datetime, timezone


load_dotenv()

def chat_with_ai_with_references(prompt: str) -> str:
    """
    Chat with the AI using OpenAI's API, using references from the vector database.

    Args:
        prompt (str): The input prompt for the AI.

    Returns:
        str: The AI's response.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Query the vector database for references
    chroma_path = os.getenv("CHROMA_PATH", ".chroma")
    collection_name = os.getenv("COLLECTION_NAME", "knowledge_summaries")

    vector_store = VectorStore(
        path=chroma_path,
        collection_name=collection_name,
        embedding_model="text-embedding-v3",
        openai_api_key=api_key,
        openai_base_url=base_url,
        embedding_dimensions=1024,
    )

    references = []
    try:
        # Ensure the prompt is wrapped in a list for the vector database query
        # Update the query method call to include only documents in the results
        results = vector_store.collection.query(query_texts=[prompt], include=["documents"])
        references = results["documents"] if "documents" in results else []
    except Exception as e:
        print(f"[chat_with_ai_with_references] Error querying vector DB: {e}")

    # Combine references into the prompt
    # Flatten the references list to ensure all items are strings
    references = [doc if isinstance(doc, str) else str(doc) for doc in references]
    references_text = "\n\n".join(references)
    #print(f"[chat_with_ai_with_references] References found: {references_text}")
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    extended_prompt = (
        f"You are a helpful assistant. Now, it is UTC {current_time}. Use the following references to answer the user's question.\n\n"
        f"References:\n{references_text}\n\n"
        f"User's question: {prompt}"
    )

    # Ensure telemetry is fully disabled by checking the environment variable
    if os.getenv("CHROMA_TELEMETRY_DISABLED") != "true":
        os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
        print("[chat_with_ai_with_references] Telemetry disabled.")

    try:
        response = client.chat.completions.create(
            model="qwen-plus",  # Replace with the desired model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": extended_prompt},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[chat_with_ai_with_references] Error: {e}")
        return "An error occurred while communicating with the AI."

if __name__ == "__main__":
    print("Welcome to the AI Chat Tool with References!")
    while True:
        user_prompt = input("Enter your prompt for the AI: ")
        if user_prompt.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = chat_with_ai_with_references(user_prompt)
        print(f"AI Response: {response}")
