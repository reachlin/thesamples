import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from study import VectorStore
from datetime import datetime, timezone


load_dotenv()

# Global variable to store chat history
chat_history = []

# Load capabilities from capabilities.json
try:
    with open('capabilities.json', 'r') as f:
        capabilities = json.load(f)
except FileNotFoundError:
    print("Warning: capabilities.json not found, starting with empty capabilities list")
    capabilities = []
except json.JSONDecodeError:
    print("Warning: Error parsing capabilities.json, starting with empty capabilities list")
    capabilities = []

# Initialize vector store globally to avoid re-initialization
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
chroma_path = os.getenv("CHROMA_PATH", ".chroma")
collection_name = os.getenv("COLLECTION_NAME", "knowledge_summaries")

vector_store = None
if api_key:
    vector_store = VectorStore(
        path=chroma_path,
        collection_name=collection_name,
        embedding_model="text-embedding-v3",
        openai_api_key=api_key,
        openai_base_url=base_url,
        embedding_dimensions=1024,
    )

def chat_with_ai_with_references(prompt: str) -> str:
    """
    Chat with the AI using OpenAI's API, using references from the vector database.

    Args:
        prompt (str): The input prompt for the AI.

    Returns:
        str: The AI's response.
    """
    global chat_history, vector_store

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Query the vector database for references
    references = []
    if vector_store:
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
    
    # Create system message with references and current time
    system_message = (
        f"You are a helpful assistant. Now, it is UTC {current_time}. Use the following references to answer the user's question.\n\n"
        f"References:\n{references_text}"
    )

    # Ensure telemetry is fully disabled by checking the environment variable
    if os.getenv("CHROMA_TELEMETRY_DISABLED") != "true":
        os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
        #print("[chat_with_ai_with_references] Telemetry disabled.")

    try:
        # Build messages array with system message, chat history, and current prompt
        messages = [{"role": "system", "content": system_message}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="qwen-plus",  # Replace with the desired model
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        
        ai_response = response.choices[0].message.content
        
        # Add current exchange to chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": ai_response})
        
        # Manage chat history length with AI summarization
        if len(chat_history) > 10:  # When we have more than 10 messages
            print(f"[chat_with_ai_with_references] Chat history reached {len(chat_history)} messages. Summarizing older conversation...")
            
            # Take the first 8 messages to summarize, keep the last 2 fresh
            messages_to_summarize = chat_history[:-2]  # All but last 2 messages
            recent_messages = chat_history[-2:]        # Keep last 2 messages
            
            # Get AI summary of older conversation
            summary = summarize_chat_history(client, messages_to_summarize)
            
            # Replace old history with summary + recent messages
            chat_history = [
                {"role": "assistant", "content": f"[Previous conversation summary]: {summary}"}
            ] + recent_messages
            
            print(f"[chat_with_ai_with_references] Summarization complete. New chat history length: {len(chat_history)} messages.")
        
        return ai_response
    except Exception as e:
        print(f"[chat_with_ai_with_references] Error: {e}")
        return "An error occurred while communicating with the AI."

def summarize_chat_history(client, chat_history_to_summarize):
    """
    Use AI to summarize the chat history when it gets too long.
    
    Args:
        client: OpenAI client instance
        chat_history_to_summarize: List of chat messages to summarize
        
    Returns:
        str: Summarized conversation context
    """
    try:
        # Create a prompt to summarize the conversation
        conversation_text = ""
        for msg in chat_history_to_summarize:
            role = msg["role"]
            content = msg["content"]
            conversation_text += f"{role.upper()}: {content}\n\n"
        
        summary_prompt = (
            "Please summarize the following conversation, preserving key information, "
            "context, and any important details that might be relevant for future responses. "
            "Keep the summary concise but comprehensive:\n\n" + conversation_text
        )
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates concise but comprehensive conversation summaries."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent summaries
            max_tokens=500,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"[summarize_chat_history] Error creating summary: {e}")
        return "Previous conversation context available but could not be summarized."

def clear_chat_history():
    """Clear the chat history."""
    global chat_history
    chat_history = []
    print("Chat history cleared.")

def evaluate_response(question, answer):
    """Evaluate the AI response and suggest next steps."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    action = {
        "action": "action name",
        "details": "action details"
    }
    capabilities = json.load(open("capabilities.json", "r"))
    prompt = f"""
        You are a helpful personal assistant. Given the user's question and a previous AI's answer, determine if any action you could take further to satisfy the user with a provided tools list.
        If there is no action needed, or the question is unclear, just return the exact string "NO_ACTION" to indicate no further action is needed.
        If you find any tool with a status ready from the tools list, return the name of that tool, followed by details on how to use it.
        If you find any tool but its status is not ready, return the exact string "WAITING" to indicate the tool is not ready.
        If you couldn't find any tool, you should call the engineer tool from the tools list to make a new tool.
        Remember, you can do anything with tools created by the engineer tool including interacting with my computer and the internet.
        Your output should be in json format like this:
        {action}
        The action list is also in a json format.
        The action list:
        {capabilities}
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"<question>{question}</question><answer>{answer}</answer>"}
    ]
    
    try:
        # ask ai with messages
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[evaluate_response] Error: {e}")
        return "NO_ACTION"


if __name__ == "__main__":
    print("Welcome to the AI Chat Tool with References!")
    print("Type 'clear' to clear chat history, or 'exit'/'quit' to quit.")
    while True:
        user_prompt = input("Enter your prompt for the AI: ")
        if user_prompt.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        elif user_prompt.lower() == "clear":
            clear_chat_history()
            continue
        response = chat_with_ai_with_references(user_prompt)
        print(f"\033[32m[AI Response]\033[0m {response}\n")
        # add a step to take further action based on the response, and user_prompt
        # call ai to take action based on the response
        action_response = "DUMMY"
        while action_response:
            action_response = evaluate_response(user_prompt, response)
            print(f"[debug][evaluate_response] {action_response}")
            if "NO_ACTION" in action_response:
                break
            if "WAITING" in action_response:
                print(f"\033[33m[AI Action]\033[0m waiting for the tool...\n")
                break
            
            # Try to parse action_response as JSON
            try:
                # remove ```json if present
                action_response = action_response.replace("```json", "").replace("```", "").strip()
                action_data = json.loads(action_response)
                if isinstance(action_data, dict) and action_data.get('action') == '000engineer':
                    print(f"\033[35m[AI Action]\033[0m Calling engineer to create a new tool...\n")

                    tool_name = f"custom_tool_{len(capabilities)}"
                    # Create a requirements file with the details
                    requirements_content = action_data.get('details', 'Create a new tool based on the current context')
                    with open(tool_name+".py", 'w') as f:
                        f.write(f"Tool Requirements:\n{requirements_content}\n")
                    
                    # Call the engineer tool
                    import subprocess
                    try:
                        # Start engineer.py without waiting for it to complete, suppress output
                        with open(os.devnull, 'w') as devnull:
                            subprocess.Popen(['python', 'engineer.py', tool_name+".py"], 
                                           stdout=devnull, stderr=devnull)
                        print(f"\033[35m[Engineer]\033[0m Engineer started in background...")
                        
                        # Create a new tool entry for capabilities
                        new_tool = {
                            "name": tool_name,
                            "description": requirements_content,
                            "status": "init",
                            "version": 0,
                            "file": tool_name+".py",
                            "usage": f"Tool created by engineer: {requirements_content}"
                        }
                        
                        # Append to capabilities list
                        capabilities.append(new_tool)
                        
                        # Update capabilities.json file
                        with open('capabilities.json', 'w') as f:
                            json.dump(capabilities, f, indent=4)
                        
                        print(f"\033[35m[Tool Created]\033[0m New tool added to capabilities: {new_tool['name']}")
                        
                    except Exception as e:
                        print(f"\033[31m[Engineer Error]\033[0m Error starting engineer: {e}")
                    
                    break
                else:
                    print(f"\033[35m[AI Action] Calling\033[0m {action_response}\n")
                    # TODO: call the tool action_data.get('action')
                    # needs LLM to extract parameters from action_data.get('details') and user_prompt
                    break
                    
            except json.JSONDecodeError:
                # If it's not valid JSON, treat it as a regular action response
                print(f"\033[31m[AI Action] Error\033[0m {action_response}\n")
                break

