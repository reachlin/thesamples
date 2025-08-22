# Water Cloud Body — Autonomous Research Agent

An autonomous Python agent that cycles through topics from `interests.json`, searches the web (DuckDuckGo), scrapes pages, summarizes findings (OpenAI-compatible or fallback), and stores summaries in a local Chroma vector database for later retrieval.

## Features
- OpenAI-compatible endpoints (supports custom `OPENAI_BASE_URL`)
- Configurable embedding dimensions (`EMBEDDING_DIMENSIONS`, default 1024)
- Local persistent Chroma vector DB in `.chroma/`
- Robust logging/debug prints across the workflow
- Graceful fallbacks for summarization and embeddings

## Requirements
- Python 3.10+
- Internet access
- pip (or conda if you prefer using `environment.yml`)

## Quickstart
1) Install dependencies
- With pip:
  - `python3 -m pip install -r requirements.txt`
- Or with conda (optional):
  - `conda env create -f environment.yml`
  - `conda activate water-cloud-body`

2) Configure environment
- Copy `.env.example` to `.env` and edit as needed:
  - `OPENAI_API_KEY` — required for model-based summaries & embeddings. If omitted, a simple fallback summarizer is used.
  - `OPENAI_BASE_URL` — optional for OpenAI-compatible endpoints (e.g., Azure OpenAI, local servers, third-party providers).
  - `EMBEDDING_DIMENSIONS` — integer, default 1024. Must match your embedding model if provider enforces a specific size.
  - `AGENT_SLEEP_SECONDS`, `MAX_DOCS_PER_TOPIC` — control cadence and depth.

3) Choose topics
- Edit `interests.json` with a list of objects like: `[ { "topic": "quantum computing" }, { "topic": "LLM routing" } ]`.

4) Run the agent
- `python3 main.py`
- The agent runs indefinitely. Press Ctrl+C to stop.

## Configuration (env vars)
- `AGENT_SLEEP_SECONDS` (default: 300) — delay between cycles
- `MAX_DOCS_PER_TOPIC` (default: 5) — max pages per topic
- `MODEL` (default: `qwen-plus`) — chat model for summarization
- `EMBEDDING_MODEL` (default: `text-embedding-v3`) — embedding model name
- `EMBEDDING_DIMENSIONS` (default: 1024) — embedding vector size
- `OPENAI_API_KEY` — API key for OpenAI-compatible endpoints
- `OPENAI_BASE_URL` — base URL for OpenAI-compatible API
- `CHROMA_PATH` (default: `.chroma`) — persistence path for vector DB
- `COLLECTION_NAME` (default: `knowledge_summaries`) — Chroma collection name

## How it works
1. Load topics from `interests.json`.
2. Search DuckDuckGo HTML results.
3. Fetch and clean page content.
4. Summarize via OpenAI-compatible chat completions (or fallback extractive summary).
5. Embed summaries and upsert into Chroma persistent store.

## Data & Storage
- Vector DB persists under `.chroma/` in the project directory.
- Each summary stores metadata: topic, title, URL, and the summary text.

## Troubleshooting
- Missing Chroma: Ensure `chromadb` installed (`pip install -r requirements.txt`).
- Embedding errors about dimensions: adjust `EMBEDDING_DIMENSIONS` to match your embedding model or remove it if provider doesn’t support custom dims.
- OpenAI errors: verify `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- Few/empty results: web pages may block scraping or have minimal text. Try different topics or wait/retry.

## Notes
- DuckDuckGo HTML scraping is best-effort and may vary.
- This agent is for research/demonstration; verify content before using summaries in production.
