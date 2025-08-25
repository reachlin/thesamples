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
    sleep_seconds: int = 60
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
                max_tokens=1000,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[summarize] OpenAI error: {e}")
            # Fallback
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]
            return f"Topic: {topic}\nTitle: {title}\nURL: {url}\nSummary: " + ". ".join(sentences[:5])

    def break_down_topic(self, topic: str, text: str) -> List[str]:
        if self.client is None:
            print(f"[debug][break_down_topic] Using fallback for topic breakdown: {topic}")
            # Fallback: Extract keywords as subtopics
            words = [word.strip() for word in text.split() if len(word.strip()) > 3]
            return list(set(words[:10]))  # Return first 10 unique words as subtopics

        prompt = (
            "You are a helpful research assistant. Break down the given topic into a list of subtopics for further study. "
            "Focus on key areas, concepts, or questions related to the topic. Provide the subtopics as a JSON array of strings."
            "\n\n"
            f"Topic: {topic}\n\nCONTENT:\n{text[:6000]}"
        )
        try:
            print(f"[debug][break_down_topic] Requesting subtopics for: {topic}")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You break down topics into subtopics for further study."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2
            )
            content = resp.choices[0].message.content
            # Clean up the AI response content
            content = content.strip()
            if "```json" in content:
                start = content.find("```json")
                end = content.rfind("```")
                if start != -1 and end != -1 and start != end:
                    content = content[start + 7:end].strip()
            else:
                if "```" in content:
                    start = content.find("```")
                    end = content.rfind("```")
                    if start != -1 and end != -1 and start != end:
                        content = content[start + 3:end].strip()
            #print(f"[debug][break_down_topic] AI response: {content}")
            try:
                # Attempt to parse as JSON
                subtopics = json.loads(content)
                if isinstance(subtopics, list) and all(isinstance(item, str) for item in subtopics):
                    return subtopics
                else:
                    print("[debug][break_down_topic] AI response is not a valid JSON list of strings")
            except json.JSONDecodeError:
                print("[debug][break_down_topic] AI response is not JSON, treating as plain string")


            #print(f"[debug][break_down_topic] Cleaned AI response: {content}")
            # Fallback: Split plain string response into lines
            return [line.strip() for line in content.split("\n") if line.strip()]
        except Exception as e:
            print(f"[break_down_topic] OpenAI error: {e}")
            return []  # Return empty list on failure


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

        # Check for similar documents
        try:
            print(f"[debug][vector] Checking for similar documents for title='{title}'")
            similar_docs = self.collection.query(
                query_texts=[summary],
                n_results=1  # Only need the most similar document
            )
            if similar_docs and similar_docs["documents"]:
                print(f"[debug][vector] Found similar document for title='{title}', skipping insert.")
                return
        except Exception as e:
            print(f"[vector] error checking for similar documents: {e}")

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
            current_time = int(time.time())
            for item in data:
                if isinstance(item, dict) and "topic" in item and isinstance(item["topic"], str):
                    last_studied = item.get("lastStudied", "")
                    if last_studied and current_time - int(last_studied) < 8 * 3600:
                        print(f"[debug][interests] Skipping topic '{item['topic']}' (studied recently)")
                        continue
                    topics.append({"topic": item["topic"].strip(), "lastStudied": last_studied})
            print(f"[debug][interests] Loaded {len(topics)} topic(s)")
            return topics
        except FileNotFoundError:
            print(f"[interests] file not found at {self.path}")
            return []
        except Exception as e:
            print(f"[interests] error loading interests: {e}")
            return []

    def update_last_studied(self, topic: str):
        print(f"[debug][interests] Updating lastStudied for topic: {topic}")
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("interests.json must be a list of {topic}")

            updated = False
            for item in data:
                if item.get("topic") == topic:
                    item["lastStudied"] = str(int(time.time()))  # Update with current Unix timestamp
                    updated = True
                    break

            if updated:
                with open(self.path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[debug][interests] Updated lastStudied for topic: {topic}")
            else:
                print(f"[debug][interests] Topic not found: {topic}")
        except Exception as e:
            print(f"[interests] error updating lastStudied: {e}")

    def append_subtopics(self, subtopics: List[str], parent: str = ""):
        print(f"[debug][interests] Appending subtopics: {subtopics} with parent: {parent}")
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("interests.json must be a list of {topic}")

            existing_topics = {item["topic"] for item in data if isinstance(item, dict) and "topic" in item}
            new_topics = [
                {"topic": subtopic.strip(), "lastStudied": "", "parent": parent}
                for subtopic in subtopics
                if subtopic.strip() not in existing_topics
            ]

            if new_topics:
                data.extend(new_topics)
                with open(self.path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[debug][interests] Added {len(new_topics)} new subtopic(s)")
            else:
                print("[debug][interests] No new subtopics to add")
        except Exception as e:
            print(f"[interests] error appending subtopics: {e}")

    def clean_invalid_topics(self):
        print("[debug][interests] Cleaning invalid topics from interests.json")
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("interests.json must be a list of {topic}")

            valid_topics = [
                item for item in data
                if isinstance(item, dict)
                and "topic" in item
                and isinstance(item["topic"], str)
                and item["topic"].strip() not in ("", "---")
            ]

            if len(valid_topics) != len(data):
                with open(self.path, "w") as f:
                    json.dump(valid_topics, f, indent=2)
                print(f"[debug][interests] Removed {len(data) - len(valid_topics)} invalid topic(s)")
            else:
                print("[debug][interests] No invalid topics found")
        except Exception as e:
            print(f"[interests] error cleaning invalid topics: {e}")

    def clean_json_tags(self):
        print("[debug][interests] Cleaning JSON tags from interests.json")
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("interests.json must be a list of {topic}")

            for item in data:
                if isinstance(item, dict) and "topic" in item and isinstance(item["topic"], str):
                    topic = item["topic"].strip()
                    if topic.startswith("```") and topic.endswith("```"):
                        item["topic"] = topic.strip("`").strip()
                    elif topic.startswith("```json") and topic.endswith("```"):
                        item["topic"] = topic[7:].strip("`").strip()

            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)
            print("[debug][interests] Cleaned JSON tags from topics")
        except Exception as e:
            print(f"[interests] error cleaning JSON tags: {e}")


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
        self.cycle_count = 0  # Initialize cycle counter
        print(
            "[debug][agent] Config -> "
            f"sleep_seconds={cfg.sleep_seconds}, max_docs_per_topic={cfg.max_docs_per_topic}, "
            f"collection='{cfg.collection_name}', chroma_path='{cfg.chroma_path}', "
            f"model='{cfg.model}', embedding_model='{cfg.embedding_model}', dims={cfg.embedding_dimensions}, "
            f"openai_base_url='{base_url or 'default'}', openai_client={'on' if self.openai_client else 'off'}"
        )

    def cycle(self):
        self.cycle_count += 1  # Increment cycle counter
        print(f"[agent] Starting cycle {self.cycle_count}")
        self.loader.clean_json_tags()  # Clean JSON tags before loading topics
        self.loader.clean_invalid_topics()  # Clean invalid topics before loading
        interests = self.loader.load()
        if not interests:
            print("[agent] No interests found. Sleeping...")
            time.sleep(self.cfg.sleep_seconds)
            return

        # Incrementally increase search result size per cycle
        #search_result_size = min(20, self.cycle_count + 4)  # Start from 5, max out at 20
        search_result_size = 3
        print(f"\n[agent] Cycle: {self.cycle_count},\nSearch Result Size: {search_result_size},\nTotal Interests: {len(interests)}\n")

        for item in interests:
            topic = item["topic"]
            print(f"[agent] Researching: {topic}")
            results = WebSearch.search(topic, search_result_size)
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

                    # Generate subtopics for further study
                    subtopics = self.summarizer.break_down_topic(topic, text)
                    if subtopics:
                        print(f"[agent] Subtopics for '{topic}': {subtopics}")
                        self.loader.append_subtopics(subtopics, parent=topic)  # Append subtopics to interests.json with parent topic
                    else:
                        print(f"[agent] No subtopics generated for '{topic}'")
            print(f"\n[debug][agent] Completed topic: {topic}\n")
            self.loader.update_last_studied(topic)  # Update last studied time

    def run_forever(self):
        print("[agent] Starting. Press Ctrl+C to stop.")
        while True:
            try:
                self.cycle()
                # Calculate sleep time based on cycle count
                dynamic_sleep = self.cfg.sleep_seconds + (self.cycle_count * 60)  # Increase sleep by 60 seconds per cycle
                print(f"\n[agent] Sleeping for {dynamic_sleep}s before next cycle...\n")
                time.sleep(dynamic_sleep)
            except KeyboardInterrupt:
                print(f"[agent] Stopping after {self.cycle_count} cycles.")
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
