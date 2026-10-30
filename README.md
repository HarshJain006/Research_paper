# Marginal — Agentic AI Research Paper Explainer

## Architecture

**Multi-agent pipeline (no LangChain)** — `agents.py` is a small hand-rolled
orchestrator. An "agent" is just an async function sharing one `AgentContext`;
concurrency is plain `asyncio.create_task` / `asyncio.gather`:

```
Planner Agent
   -> title, field, search queries, checkable claims (1 quick JSON call)
Research Agent  (runs in the background while sections A/B/C/takeaways start)
   -> queries run concurrently against:
        - web search (Tavily if TAVILY_API_KEY set, else DuckDuckGo, else MCP)
        - Semantic Scholar  (real "how many papers match" counts)
        - arXiv             (independent count + recent preprints)
Section-Writer Agents (6 total, each its OWN independent LLM call/context window)
   -> A: Executive Summary / Technical / Layman           — starts immediately
   -> B: Innovation / Author's Thinking / Math             — starts immediately
   -> C: Results / Strengths & Limitations / Applications  — starts immediately
   -> takeaways: Key Takeaways                             — starts immediately
   -> related: Related Work & Novelty Check                — waits for Research
   -> ideas: Research Paper Ideas (the main "help me write my own paper" section) — waits for Research
Assembler
   -> concatenates sections + a programmatic "Sources Consulted" section
```

Every section call is fully independent — its own system prompt, its own
context window, no dependency on another section's output — so it can't blow
past a small local model's context limit. The only thing shared across calls
is a two-line anchor (paper title + field, from the Planner) injected into
every prompt, which keeps voice/framing consistent without coupling the calls.

**Live streaming, end to end.** Both `/api/analyze` and `/api/chat` are
Server-Sent Events endpoints — you watch each agent's tokens land in real
time rather than waiting for one big blocking response. This matters most if
`text_model` is a small local model (Gemma 3 1B, etc.) where generation is
slow enough to want to actually see it happen.

`/api/analyze` event types (`data: {...}\n\n` frames, ending in `data: [DONE]`):

| type | payload | meaning |
|---|---|---|
| `status` | `{message}` | planning / research / assembly phase updates |
| `section_start` | `{key}` | one of the 6 section agents began |
| `section_delta` | `{key, delta}` | a streamed token/chunk from that agent |
| `section_done` | `{key}` | that agent finished |
| `research_done` | `{queries, related_papers_found, used_mcp}` | research phase complete |
| `final` | `{report_markdown, figures, meta, session_id}` | fully assembled + ready to render |
| `error` | `{message}` | something failed |

`/api/chat/stream` streams the same way: `data: {"delta": "..."}` chunks,
ending in `data: [DONE]`. (`/api/chat` also still exists as a plain
non-streaming JSON endpoint if you need it.)

**MCP support** — `mcp_client.py` is a real MCP (Model Context Protocol)
client. Set `MCP_SERVER_URL` to any streamable-HTTP MCP server that exposes
a search tool (e.g. a hosted Tavily/Exa/Brave MCP endpoint) and the Research
Agent calls it over the actual protocol. If unset, or the `mcp` package isn't
installed, or the call fails, everything falls back automatically to the
plain REST tools in `research_tools.py` — nothing else depends on MCP.

**"How many such papers are out there"** — answered by Semantic Scholar's
and arXiv's own match counts (`research_tools.py`), not an LLM guess. The
Research Agent surfaces both per query; the Ideas section uses them as a
real novelty signal (e.g. "~12 S2 hits, ~4 arXiv — underexplored").

**RAG chat sidebar** — `rag.py`. Every analyzed paper is chunked (~900
chars, 150 overlap) and indexed with BM25 (`rank_bm25`, no vector DB needed
for one document). The "💬 Ask this paper" button opens a sliding sidebar;
each question retrieves the top-5 relevant chunks and streams an answer
grounded only in those chunks.

**Prompts kept small on purpose.** The shared formatting rules and each
section's own instructions were trimmed hard (short sentences, one
instruction per line) so they cost little context and small models can
actually follow them — full prompt text lives in `agents.py` if you want to
tune further. `MAX_PAPER_CHARS` and `MAX_OUTPUT_TOKENS` are both env-
configurable (`.env.example`) if you need to shrink the per-call budget
further for a very small local model.

**Visuals** — the "Research Paper Ideas" and "Related Work" sections get
distinct highlighted styling. Section agents may emit one `mermaid` diagram
each when it genuinely clarifies an architecture/flow, rendered client-side
as real SVG via mermaid.js. Math is written as real LaTeX (`$...$` /
`$$...$$`) and rendered client-side with KaTeX — the frontend protects math
spans from Markdown's parser before rendering, then restores and renders
them, so subscripts/braces in equations don't get mangled.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in GROQ_API_KEY (required), or point at a local OpenAI-compatible
# server (e.g. llama.cpp / Ollama / LM Studio) via the "custom provider"
# fields in the UI for something like Gemma 3 1B
# optionally: TAVILY_API_KEY (better web search) and/or MCP_SERVER_URL
uvicorn main:app --reload --port 8000
```

Open `index.html` directly in a browser (or serve it statically) and point
"Backend URL" at `http://localhost:8000`.

Nothing external is required to get *some* research signal — Semantic
Scholar and arXiv need no API key, and the DuckDuckGo web-search fallback
needs no key either. Tavily/MCP are optional quality upgrades.

## Deploy on Streamlit Community Cloud

The Streamlit app displays the existing `index.html` frontend. The FastAPI
backend must also be deployed to a public Python host; Streamlit Community
Cloud does not expose this project's FastAPI service. The browser calls that
backend directly, including its streaming analysis and chat endpoints.

1. Deploy this folder's `main.py` as a FastAPI web service. Install
   `requirements.txt` and start it with:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

   Set `GROQ_API_KEY` in the backend service's environment. Optionally set
   `TAVILY_API_KEY`, `MCP_SERVER_URL`, and `MCP_SEARCH_TOOL_NAME` there too.
2. Push this project folder to GitHub.
3. In Streamlit Community Cloud, create an app from that repository and set
   **Main file path** to `streamlit_app.py` (or its path relative to the
   repository root).
4. In the Streamlit app's **Settings → Secrets**, set the URL of the deployed
   FastAPI service:

   ```toml
   BACKEND_URL = "https://your-backend.example.com"
   ```

   The frontend uses this URL by default; it can also be edited in its
   **Backend URL** settings. Do not put `GROQ_API_KEY` in Streamlit secrets or
   the browser when it is already configured on the backend.

The deployed backend currently has no user authentication or rate limiting.
Restrict access or add an authentication/rate-limiting layer before sharing
it publicly, to prevent unauthorized use of its configured model API key.

## Files

- `streamlit_app.py` — Streamlit Cloud entry point that embeds the frontend.
- `main.py` — FastAPI app: PDF ingestion, figure extraction, the two SSE
  endpoints (`/api/analyze`, `/api/chat/stream`), plus a plain `/api/chat`.
- `agents.py` — the multi-agent orchestrator (Planner, Research, 6 Section
  Writers, Assembler) and every system prompt.
- `research_tools.py` — Tavily/DuckDuckGo web search, Semantic Scholar,
  arXiv. All fail soft (return empty results instead of raising).
- `mcp_client.py` — optional real MCP client, fails soft if unconfigured.
- `rag.py` — BM25 chunking/retrieval + in-memory session store for chat.
- `index.html` — the whole frontend (vanilla JS, no build step): upload UI,
  live streaming section panels, final report renderer (Markdown + KaTeX +
  Mermaid), and the RAG chat sidebar.
- `requirements.txt`, `.env.example` — dependencies and configuration.

## Notes / things you may want to extend

- Sessions (for the RAG chat) are in-memory and expire after 6h of
  inactivity — fine for local/single-user use; swap for Redis if you
  deploy this for multiple people at once.
- Each section agent still receives the *full* paper text as input (up to
  `MAX_PAPER_CHARS`), just as its own independent call — that's what keeps
  output length bounded per model, but a very long paper is still a lot of
  *input* tokens per call for a tiny local model. If you need to go further,
  the natural next step is retrieval-based input (only feed each section the
  paper chunks relevant to it) rather than the whole text — `rag.py`'s BM25
  index could be reused for that.