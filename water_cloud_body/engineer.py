import time
from dotenv import load_dotenv
from openai import OpenAI
import os
import sys
from sys import argv

# AI agent to improve a python script based on its output
# Load environment variables from .env file
# Usage: python engineer.py [script_file] [output_file]

load_dotenv()

MAX_ROUNDS = 1024  # Maximum number of iterations for AI to improve the code
SCRIPT_FILE = './search.py'
OUTPUT_FILE = './output.log'

AI_PROMPT = """
You are a helpful python code assistant.
You will be given a python script, enclosed in <code> and </code>, and its output enclosed in <output> and </output>.
Generally, the given script should have its purpose clearly described in comments at the top.
Your job is to analyze the script and its output, and fix any issues found in the output.
Your response is either:
The new script fixed and enclosed in triple backticks, if you find any issues in the script.
Or, just return this exact message "PERFECT CODE" to indicate no changes are needed, if you find no issues in the script.
"""
def ask_ai(question):
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        completion = client.chat.completions.create(
            model="qwen-plus",  # https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': AI_PROMPT},
                {'role': 'user', 'content': question}
                ]
        )
        rtn = completion.choices[0].message.content
        print(f"AI response: {rtn}")
        return rtn
    except Exception as e:
        print(f"Error：{e}")

if __name__ == "__main__":
    print("[info] Engineer wakes up to work")
    round = 0
    while True:
        round += 1
        try:
            SCRIPT_FILE = argv[1] if len(argv) > 1 else SCRIPT_FILE
            OUTPUT_FILE = argv[2] if len(argv) > 2 else OUTPUT_FILE
            # run the script file and capture its output
            print(f"[info] ({round})Running the script {SCRIPT_FILE}...")
            os.system(f"python {SCRIPT_FILE} > {OUTPUT_FILE} 2>&1")
        except Exception as e:
            print(f"[error] Error during script execution: {e}")
        time.sleep(3)
        # read the script and its output from output.log
        f = open(SCRIPT_FILE, 'r')
        code = f.read()
        f.close()
        f = open(OUTPUT_FILE, 'r')
        output = f.read()
        f.close()
        improved_code = ask_ai(f"Please analyze the following code and its output:\n\n<code>{code}</code>\n\n<output>{output}</output>\n\nFix the errors and improve the code.")
        if "PERFECT CODE" in improved_code:
            print("[info] AI thinks the code is perfect, no changes needed.")
            break
        if  round > MAX_ROUNDS:
            print("[info] AI has reached the maximum number of iterations.")
            break
        # write the improved code back to the script file
        with open(SCRIPT_FILE, 'w') as f:
            # Find the code block in the improved response,
            # and write it back to search.py.
            # The code block is inside triple backticks, with some additional text before and after.
            # find the starting and ending triple backticks
            start_length = len('```python')
            start = improved_code.find('```python')
            if start == -1:
                start = improved_code.find('```')
                start_length = len('```')
            end = improved_code.find('```', start + start_length)
            if start != -1 and end != -1 and end > start:
                improved_code = improved_code[start + start_length:end].strip()
                f.write(improved_code)
                print(f"[debug] Improved code written.")
            f.close()
        # Restart the script to apply changes, if the user inputs 'y'
        #if input("Do you want to restart the script? (y/n): ").strip().lower() == 'y':
        time.sleep(3)