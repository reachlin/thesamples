import datetime
import random
import pyautogui

from PIL import Image
from paddleocr import PaddleOCR

import pyautogui
import requests
from mcp.server.fastmcp import FastMCP

# Create server
mcp = FastMCP("Echo Server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print(f"[debug-server] add({a}, {b})")
    return a + b


@mcp.tool()
def get_secret_word() -> str:
    print("[debug-server] get_secret_word()")
    return random.choice(["apple", "banana", "cherry"])


@mcp.tool()
def get_current_weather(city: str) -> str:
    print(f"[debug-server] get_current_weather({city})")

    endpoint = "https://wttr.in"
    response = requests.get(f"{endpoint}/{city}")
    return response.text

def screenshot() -> str:
    """Take a screenshot and save it to a file"""
    print(f"[debug-server] screenshot()")
    screenshot = pyautogui.screenshot()
    saved_file = f"screenshot{int(datetime.datetime.now(datetime.timezone.utc).timestamp())}.png"
    screenshot.save(saved_file)
    return saved_file

def scan_image(image):
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        show_log=False,
        # det_max_side_len=2000,
        # max_text_length=200,
        ocr_version="PP-OCRv4"
    )  # need to run only once to download and load model into memory
    result = ocr.ocr(image, cls=True)
    # for idx in range(len(result)):
    #     res = result[idx]
    #     for line in res:
    #         print(line)
    return result

import re
DEBUG = True
g_document = []

def find_tokens(to_find, after=None):
    print(f"[debug-server] find_tokens({to_find}, {after}, document_len={len(g_document)})")
    if not g_document:
        return None
    index = 0
    for item in g_document:
        text, score = item[1]
        if DEBUG:
            print(f"{score} {text}")
        if score > 0.8:
            match = re.search(to_find, text)
            if match:
                print("Match Found: ", index, item)
                p0x, p0y = item[0][0]
                p1x, p1y = item[0][1]
                p2x, p2y = item[0][2]
                length_text = len(text)
                bp, ep = match.span()
                x = p0x + (p1x - p0x)*(bp+1)/length_text
                y = p0y + (p2y - p0y)/3
                if after:
                    if index > after:
                        return index, x, y, match.group()
                else:
                    return index, x, y, match.group()
        index += 1
    return None

@mcp.tool()
def find_text_on_screen(to_find: str) -> str:
    """find text on the screen and return its position"""
    print(f"[debug-server] find_text_on_screen({to_find})")
    rtn = find_tokens(to_find)
    if rtn:
        index, x, y, text = rtn
        print(f"{index} {int(x)} {int(y)} {text}")
        return f"x: {int(x)}, y: {int(y)}, width: {g_width}, height: {g_height}"
    return "not found"

@mcp.tool()
def read_screenshot() -> str:
    """Take a screenshot and read text from it"""
    print(f"[debug-server] read_screenshot()")
    saved_file = screenshot()
    image = Image.open(saved_file)
    global g_width, g_height
    g_width, g_height = image.size
    print(f"[debug-server] read_screenshot() - image size: {g_width} x {g_height}")
    
    # send to google for OCR to find the key word
    #pyautogui.moveTo(30, 30, 3)
    document = scan_image(saved_file)
    if document and len(document)>0:
        print(f"[debug-server] read_screenshot() - found: {document[0]}")
        global g_document
        g_document = document[0]
        return f"{document[0]}"
    return "nothing readable"

if __name__ == "__main__":
    mcp.run(transport="sse")

    