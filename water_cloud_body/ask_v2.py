# Demo. the usage of my local screen OCR MCP.
from agents.mcp import MCPServer, MCPServerSse
from agents import Agent, Runner, function_tool, OpenAIConversationsSession
from agents.model_settings import ModelSettings
import pyautogui
import asyncio
from PIL import Image
from dotenv import load_dotenv



@function_tool
def move_mouse_to(x: int, y: int, width: int, height: int):
    """Move mouse to the position (x, y) on an image of size (width, height)"""
    sw, sh = pyautogui.size()
    cx = x*sw/width
    cy = y*sh/height
    print(f"move_mouse_to: {x}, {y} / {width}, {height} -> {cx}, {cy} / {sw}, {sh}")
    pyautogui.moveTo(cx, cy, 2)

async def main():
    print("Welcome to the AI Chat Tool with References!")
    print("Type 'clear' to clear chat history, or 'exit'/'quit' to quit.")
    load_dotenv()
    session = OpenAIConversationsSession()
    mcp_server = MCPServerSse(
        name="My SSE Python Server",
        client_session_timeout_seconds=100,
        params={
            "url": "http://localhost:8000/sse",
            # "timeout": 100.0,
            # "sse_read_timeout": 100.0,
        },
    )
    await mcp_server.connect()
        
    agent = Agent(
            name="Assistant",
            instructions="Use the tools to answer the questions.",
            mcp_servers=[mcp_server],
            model_settings=ModelSettings(tool_choice="required"),
            tools = [move_mouse_to],
        )

    while True:
        user_prompt = input("\033[32mEnter your prompt for the AI: \033[0m")
        if user_prompt.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        elif user_prompt.lower() == "clear":
            session = OpenAIConversationsSession()
            continue

        #q = "take a screenshot and summarize its contents. the screenshot will be a big array and each item is a set of coordinates followed by text recognized."
        #q = "then, find 'File' menu on my screen and move my mouse to it."
        #q = "move my mouse to google_ocr.ipynb on my screen"

        result = await Runner.run(agent, 
                                  session=session,
                                  input=user_prompt)
        print(result.final_output)
    await mcp_server.cleanup()

if __name__ == "__main__":
    asyncio.run(main())