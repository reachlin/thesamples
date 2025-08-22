import os
import json
import time
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from search import WebSearch  # Assuming this is the custom search module
from chromadb.config import Settings

try:
    import chromadb  # type: ignore
except Exception:
    chromadb = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


@dataclass
class Config:
    sleep_seconds: int = 300
    max_docs_per_topic: int = 5
    collection_name: str = "knowledge_summaries"
    chroma_path: str = ".chroma"
    model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024

class WebScraper:
    @staticmethod
    def fetch(url: str) -> Optional[str]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/117.0"
        }
        print(f"[debug][scrape] Fetching URL: {url}")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print(f"[scrape] error fetching {url}: {e}")
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = "\n".join(t.strip() for t in soup.get_text("\n").splitlines() if t.strip())
        if text:
            print(f"[debug][scrape] Extracted {len(text)} chars from {url}")
        return text[:20000]  # cap


class Summarizer:
    def __init__(self, model: str, openai_client: Optional[Any]):
        self.model = model
        self.client = openai_client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type(Exception))
    def summarize(self, topic: str, title: str, url: str, text: str) -> str:
        if self.client is None:
            print(f"[debug][summarize] Using fallback summarizer for: {title}")
            # Fallback simple extractive summary: take first few sentences.
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]
            return f"Topic: {topic}\nTitle: {title}\nURL: {url}\nSummary: " + ". ".join(sentences[:5])

        prompt = (
            "You are a helpful research assistant. Read the provided page content and write a concise, factual, neutral summary (120-180 words). "
            "Include key definitions, notable facts, and practical insights. Avoid fluff. If content is low-quality or off-topic, say so briefly.\n\n"
            f"Topic: {topic}\nTitle: {title}\nURL: {url}\n\nCONTENT:\n{text[:6000]}"
        )
        try:
            print(f"[debug][summarize] Requesting summary from model '{self.model}' for: {title}")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You summarize web pages for a knowledge base."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[summarize] OpenAI error: {e}")
            # Fallback
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]
            return f"Topic: {topic}\nTitle: {title}\nURL: {url}\nSummary: " + ". ".join(sentences[:5])


class VectorStore:
    def __init__(self, path: str, collection_name: str, embedding_model: str, openai_api_key: Optional[str], openai_base_url: Optional[str] = None, embedding_dimensions: Optional[int] = None):
        print(f"[debug][vector] Initializing Chroma at '{path}', collection='{collection_name}'")
        if chromadb is None:
            raise RuntimeError("ChromaDB is not installed. Please install 'chromadb' as specified in requirements/environment.")
        self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        embed_fn = None
        if openai_api_key and OpenAI is not None:
            print(f"[debug][vector] Using OpenAI-compatible embeddings: model='{embedding_model}', dims={embedding_dimensions}, base_url='{openai_base_url or 'default'}'")
            # Build a custom OpenAI embedding function that honors base_url and dimensions
            class _OpenAIEmbedder:
                def __init__(self, api_key: str, base_url: Optional[str], model_name: str, dimensions: Optional[int]):
                    kwargs = {"api_key": api_key}
                    if base_url:
                        kwargs["base_url"] = base_url
                    self.client = OpenAI(**kwargs)
                    self.model = model_name
                    self.dimensions = dimensions

                def __call__(self, input: List[str]):  # Chroma expects parameter named 'input'
                    try:
                        kwargs = {"model": self.model, "input": input}
                        if self.dimensions:
                            kwargs["dimensions"] = self.dimensions
                        print(f"[debug][vector] Embedding {len(input)} item(s) with model='{self.model}', dims={self.dimensions}")
                        resp = self.client.embeddings.create(**kwargs)
                    except Exception as e:
                        print(f"[vector] embedding error with dims specified ({self.dimensions}): {e}. Retrying without dimensions...")
                        # Retry without dimensions if provider/model doesn't support it
                        resp = self.client.embeddings.create(model=self.model, input=input)
                    return [d.embedding for d in resp.data]

            embed_fn = _OpenAIEmbedder(openai_api_key, openai_base_url, embedding_model, embedding_dimensions)
        else:
            # Fallback to default (no embeddings configured by provider)
            print("[debug][vector] No embedding function configured (using default).")
            embed_fn = None  # type: ignore
        self.collection = self.client.get_or_create_collection(collection_name, embedding_function=embed_fn)

    def add_summary(self, topic: str, title: str, url: str, summary: str):
        uid = hashlib.sha256(f"{url}|{title}".encode()).hexdigest()
        metadata = {"topic": topic, "title": title, "url": url}
        try:
            print(f"[debug][vector] Adding summary id={uid[:8]}... topic='{topic}', title='{title}'")
            self.collection.add(ids=[uid], documents=[summary], metadatas=[metadata])
        except Exception:
            print(f"[debug][vector] Updating existing summary id={uid[:8]} for title='{title}'")
            # Upsert behavior
            self.collection.update(ids=[uid], documents=[summary], metadatas=[metadata])


class InterestLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Dict[str, str]]:
        print(f"[debug][interests] Loading interests from: {self.path}")
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("interests.json must be a list of {topic}")
            topics = []
            for item in data:
                if isinstance(item, dict) and "topic" in item and isinstance(item["topic"], str):
                    topics.append({"topic": item["topic"].strip()})
            print(f"[debug][interests] Loaded {len(topics)} topic(s)")
            return topics
        except FileNotFoundError:
            print(f"[interests] file not found at {self.path}")
            return []
        except Exception as e:
            print(f"[interests] error loading interests: {e}")
            return []


class Agent:
    def __init__(self, cfg: Config, interests_path: str):
        load_dotenv()
        self.cfg = cfg
        self.interests_path = interests_path
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url) if (OpenAI and api_key) else None
        self.vector_store = VectorStore(cfg.chroma_path, cfg.collection_name, cfg.embedding_model, api_key, base_url, cfg.embedding_dimensions)
        self.summarizer = Summarizer(cfg.model, self.openai_client)
        self.loader = InterestLoader(interests_path)
        print(
            "[debug][agent] Config -> "
            f"sleep_seconds={cfg.sleep_seconds}, max_docs_per_topic={cfg.max_docs_per_topic}, "
            f"collection='{cfg.collection_name}', chroma_path='{cfg.chroma_path}', "
            f"model='{cfg.model}', embedding_model='{cfg.embedding_model}', dims={cfg.embedding_dimensions}, "
            f"openai_base_url='{base_url or 'default'}', openai_client={'on' if self.openai_client else 'off'}"
        )

    def cycle(self):
        interests = self.loader.load()
        if not interests:
            print("[agent] No interests found. Sleeping...")
            time.sleep(self.cfg.sleep_seconds)
            return

        for item in interests:
            topic = item["topic"]
            print(f"[agent] Researching: {topic}")
            results = WebSearch.search(topic, self.cfg.max_docs_per_topic)
            if not results:
                print(f"[agent] No search results for {topic}")
                continue
            for title, url in results:
                print(f"[debug][agent] Fetching content for: '{title}' -> {url}")
                text = WebScraper.fetch(url)
                if not text or len(text) < 300:
                    print(f"[debug][agent] Skipping '{title}' (insufficient content: {0 if not text else len(text)} chars)")
                    continue
                summary = self.summarizer.summarize(topic, title, url, text)
                if summary:
                    self.vector_store.add_summary(topic, title, url, summary)
                    print(f"[agent] Saved summary for: {title}")
            print(f"[debug][agent] Completed topic: {topic}")

    def run_forever(self):
        print("[agent] Starting. Press Ctrl+C to stop.")
        while True:
            try:
                self.cycle()
                print(f"[debug][agent] Sleeping for {self.cfg.sleep_seconds}s before next cycle...")
                time.sleep(self.cfg.sleep_seconds)
            except KeyboardInterrupt:
                print("[agent] Stopping.")
                break
            except Exception as e:
                print(f"[agent] error: {e}")
                time.sleep(self.cfg.sleep_seconds)


def load_config() -> Config:
    load_dotenv()
    sleep_seconds = int(os.getenv("AGENT_SLEEP_SECONDS", "300"))
    max_docs = int(os.getenv("MAX_DOCS_PER_TOPIC", "5"))
    embedding_dims = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    return Config(sleep_seconds=sleep_seconds, max_docs_per_topic=max_docs, embedding_dimensions=embedding_dims)


if __name__ == "__main__":
    cfg = load_config()
    agent = Agent(cfg, interests_path=os.path.join(os.path.dirname(__file__), "interests.json"))
    agent.run_forever()
