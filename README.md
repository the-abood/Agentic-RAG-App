# Agentic RAG App: LangChain + FAISS + Tavily + CrewAI

An end-to-end LLM project that combines:

- **Retrieval-Augmented Generation (RAG)** over a custom document folder, using FAISS and local Hugging Face embeddings
- **Real-time web search** with Tavily
- **Multi-agent orchestration** with CrewAI (Researcher, Writer, Critic, Trend Analyst, Report Generator), with the LLM served by Groq

Everything is in [`analysis.ipynb`](analysis.ipynb).

## What the notebook covers

| Section | Content |
|---|---|
| Base pipeline | Load `crew_data/`, chunk with `RecursiveCharacterTextSplitter`, embed, index with FAISS, `RetrievalQA`, Tavily search |
| Task 1 | `similarity_search` / `similarity_search_with_score`, PCA and t-SNE visualisation of chunk embeddings |
| Task 2 | Add new documents, rebuild and save the FAISS index, test retrieval with new queries |
| Task 3 | New **Trend Analyst** agent (Tavily tool) and a Researcher, Writer, Critic crew |
| Task 4 | **Report Generator** agent that exports a structured report to `reports/` as Markdown and PDF |

## Project structure

```
.
├── analysis.ipynb        # main notebook
├── crew_data/            # knowledge base documents (.txt)
├── reports/              # generated reports (.md / .pdf)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

Requires **Python 3.10 to 3.13**. CrewAI does not support Python 3.14, so check with `python --version`.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Option A: conda
conda create -n rag python=3.11 -y
conda activate rag

# Option B: venv (needs a 3.10-3.13 interpreter)
# python3.11 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### API keys

You need two free keys:

- **Groq**: https://console.groq.com (LLM inference)
- **Tavily**: https://tavily.com (web search)

Either export them (`export GROQ_API_KEY=...`) or just run the notebook, which prompts for them. No OpenAI key is required, since embeddings run locally with `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB, downloaded on first run).

## Run

1. Open `analysis.ipynb` and select the `rag` environment as the kernel.
2. Run the cells from the top. The first run downloads the embedding model.
3. Generated reports appear in `reports/`.

## Notes and troubleshooting

- **Model name:** Groq retires models often. The notebook prints the models your key can use; set `GROQ_MODEL` to one of them (default `openai/gpt-oss-120b`).
- **Groq rate limit (413 / 429):** the free tier has a tokens-per-minute cap. Wait a minute and re-run, or try `openai/gpt-oss-20b`. Tool outputs are already truncated to stay under the limit.
- **Event loop error:** notebooks already run an event loop, so crews are started with `await crew.kickoff_async()`. In a plain `.py` script use `crew.kickoff()`.
- **Embeddings:** the assignment mentions OpenAI Embeddings. This project uses a local Hugging Face model instead to avoid API costs. To switch back, replace `HuggingFaceEmbeddings(...)` with `OpenAIEmbeddings()` and set `OPENAI_API_KEY`.

## Possible extensions

Swap FAISS for Chroma, test other chunk sizes and embedding models, run local LLMs with Ollama, or wrap the crew in a Flask API or Streamlit app and Dockerise it.
