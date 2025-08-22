from bs4 import BeautifulSoup
import requests
from typing import List, Tuple, Optional
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebSearch")

class WebSearch:
    """Very simple web search using DuckDuckGo HTML results (no API key)."""

    DDG_URL = "https://html.duckduckgo.com/html/"
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    ]
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    RESULT_SELECTOR = "a.result__a"

    def __init__(self, max_retries: int = MAX_RETRIES, retry_delay: float = RETRY_DELAY):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @staticmethod
    def _get_random_user_agent() -> str:
        """Returns a random user agent from the list."""
        return random.choice(WebSearch.USER_AGENTS)

    @staticmethod
    def _extract_real_url(redirect_url: str) -> Optional[str]:
        """Extract the real URL from DuckDuckGo's redirect URL."""
        try:
            parsed = urlparse(redirect_url)
            if parsed.netloc == "duckduckgo.com" and parsed.path == "/l/":
                query_params = parse_qs(parsed.query)
                if "uddg" in query_params:
                    return unquote(query_params["uddg"][0])
            elif redirect_url.startswith("http"):
                return redirect_url
        except Exception as e:
            logger.warning(f"Failed to parse URL: {redirect_url} - {e}")
        return None

    @staticmethod
    def search(query: str, max_results: int = 5) -> List[Tuple[str, str]]:
        """
        Perform a DuckDuckGo search and return titles and links.

        Args:
            query (str): Search query.
            max_results (int): Maximum number of results to return.

        Returns:
            List[Tuple[str, str]]: List of (title, url) results.
        """
        logger.debug(f"Issuing search: '{query}', max_results={max_results}")
        encoded_query = quote_plus(query)
        search_url = f"{WebSearch.DDG_URL}?q={encoded_query}"

        resp = None
        for attempt in range(WebSearch.MAX_RETRIES):
            try:
                headers = {"User-Agent": WebSearch._get_random_user_agent()}
                resp = requests.get(search_url, headers=headers, timeout=WebSearch.REQUEST_TIMEOUT)
                resp.raise_for_status()
                time.sleep(random.uniform(0.5, 1.5))  # Throttle requests
                break
            except requests.RequestException as e:
                logger.error(f"Error (attempt {attempt + 1}/{WebSearch.MAX_RETRIES}) for query '{query}': {e}")
                if attempt < WebSearch.MAX_RETRIES - 1:
                    delay = (WebSearch.RETRY_DELAY * (attempt + 1)) + random.uniform(0, 1)
                    logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    return []

        # Check for CAPTCHA or unexpected content
        if "Please verify you're a human" in resp.text:
            logger.warning("CAPTCHA triggered. Try again later.")
            return []

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.error(f"Failed to parse HTML response: {e}")
            return []

        results = []

        for a in soup.select(WebSearch.RESULT_SELECTOR):
            title = a.get_text(strip=True)
            redirect_url = a.get("href")
            if not redirect_url or not title:
                continue

            actual_url = WebSearch._extract_real_url(redirect_url)
            if actual_url:
                results.append((title, actual_url))

            if len(results) >= max_results:
                break

        logger.info(f"Found {len(results)} results for '{query}'")
        return results

if __name__ == "__main__":
    # Example usage
    query = "reachlin"
    results = WebSearch.search(query, max_results=5)
    for title, url in results:
        print(f"Title: {title}\nURL: {url}\n")
    assert len(results) > 0, "No results found for the query."