"""
AI Research Paper Explainer — Backend (agentic)
------------------------------------------------
FastAPI service that:
  1. Ingests a paper as an uploaded PDF or a URL (arXiv abs/pdf link,
     huggingface.co/papers/<id>, or a direct .pdf link / a page that
     links to a citation_pdf_url).
  2. Extracts full text + embedded figures/graphs with PyMuPDF.
  3. Optionally asks a vision-capable model to describe each figure.
  4. Runs a small multi-agent pipeline (agents.py, no LangChain):
       Planner -> Research (live web + Semantic Scholar + arXiv, and MCP
       if configured) -> six parallel Section-Writer agents -> Assembler.
     This produces a 13-section report that includes a Related Work &
     Novelty Check and a dedicated Research Paper Ideas section, backed
     by real search-hit counts instead of guesses.
  5. Chunks the paper into a BM25 index (rag.py) so the frontend's
     sidebar chatbot can answer follow-up questions grounded only in
     the paper (RAG), via /api/chat.
  6. Returns structured JSON: { report_markdown, figures[], meta, session_id }.

Run:
    pip install -r requirements.txt
    cp .env.example .env   # fill in GROQ_API_KEY, optionally TAVILY_API_KEY
                            # and/or MCP_SERVER_URL for research
    uvicorn main:app --reload --port 8000
"""

import base64
import os
import re
from typing import Optional

import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

import rag
from agents import AgentContext, run_pipeline_streaming

load_dotenv()

app = FastAPI(title="AI Research Paper Explainer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_PAPER_CHARS = int(os.getenv("MAX_PAPER_CHARS", "55000"))    # per-section input budget; lower for small-context local models
MAX_FIGURES = 14               # cap figure extraction/description cost
MIN_FIGURE_DIM = 140           # px — filters out tiny icons/logos/rules
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "850"))  # per section-agent call (6 run concurrently)

FIGURE_DESC_PROMPT = (
    "This image is a figure extracted from a research paper (a chart, graph, diagram, "
    "architecture drawing, or table-as-image). In 2-3 sentences, describe exactly what it "
    "shows (axes, trend, components, or flow) and what point it is making. Be concrete, "
    "not generic."
)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class Figure(BaseModel):
    index: int
    page: int
    mime: str
    data_base64: str
    description: Optional[str] = None


class AnalyzeMeta(BaseModel):
    source: str
    pages: int
    truncated: bool
    figures_found: int
    figures_described: int
    text_model: str
    vision_model: Optional[str] = None
    research_queries: list[str] = []
    related_papers_found: int = 0
    used_mcp: bool = False


class AnalyzeResponse(BaseModel):
    report_markdown: str
    figures: list[Figure]
    meta: AnalyzeMeta
    session_id: str  # use this to chat about the paper via /api/chat


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: list[ChatMessage] = []
    provider: str = "groq"
    groq_api_key: Optional[str] = None
    groq_text_model: Optional[str] = None
    custom_base_url: Optional[str] = None
    custom_api_key: Optional[str] = None
    custom_text_model: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


# --------------------------------------------------------------------------
# PDF resolution (URL -> pdf bytes)
# --------------------------------------------------------------------------

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
UA = {"User-Agent": "Mozilla/5.0 (compatible; PaperExplainerBot/1.0)"}


async def resolve_pdf_bytes_from_url(url: str) -> bytes:
    url = url.strip()

    async with httpx.AsyncClient(headers=UA, follow_redirects=True, timeout=45) as client:
        # 1. Direct pdf link
        if url.lower().endswith(".pdf"):
            r = await client.get(url)
            r.raise_for_status()
            return r.content

        # 2. arXiv abstract page -> pdf
        if "arxiv.org/abs/" in url:
            m = ARXIV_ID_RE.search(url)
            if m:
                pdf_url = f"https://arxiv.org/pdf/{m.group(1)}.pdf"
                r = await client.get(pdf_url)
                r.raise_for_status()
                return r.content

        # 3. huggingface.co/papers/<arxiv-id>
        if "huggingface.co/papers/" in url:
            m = ARXIV_ID_RE.search(url)
            if m:
                pdf_url = f"https://arxiv.org/pdf/{m.group(1)}.pdf"
                r = await client.get(pdf_url)
                if r.status_code == 200:
                    return r.content

        # 4. Generic page: fetch HTML and look for a citation_pdf_url meta tag
        #    or the first link ending in .pdf
        r = await client.get(url)
        r.raise_for_status()
        html = r.text

        meta_match = re.search(
            r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        )
        if not meta_match:
            meta_match = re.search(
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
                html,
                re.IGNORECASE,
            )
        if meta_match:
            pdf_url = meta_match.group(1)
            r2 = await client.get(pdf_url)
            r2.raise_for_status()
            return r2.content

        href_match = re.search(r'href=["\']([^"\']+\.pdf)["\']', html, re.IGNORECASE)
        if href_match:
            pdf_url = httpx.URL(url).join(href_match.group(1))
            r2 = await client.get(str(pdf_url))
            r2.raise_for_status()
            return r2.content

    raise HTTPException(
        status_code=400,
        detail=(
            "Could not locate a downloadable PDF at that URL. "
            "Try pasting a direct arXiv link (…/abs/xxxx.xxxxx or …/pdf/xxxx.xxxxx.pdf), "
            "a huggingface.co/papers/xxxx.xxxxx link, or a direct .pdf URL."
        ),
    )


# --------------------------------------------------------------------------
# PDF parsing: text + figures
# --------------------------------------------------------------------------

def extract_text_and_figures(pdf_bytes: bytes) -> tuple[str, list[Figure], int, bool]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text_parts: list[str] = []
    for page in doc:
        text_parts.append(page.get_text("text"))
    full_text = "\n".join(text_parts).strip()
    truncated = len(full_text) > MAX_PAPER_CHARS
    rag_text = full_text  # keep the untruncated version for the RAG chat index
    if truncated:
        full_text = full_text[:MAX_PAPER_CHARS]

    figures: list[Figure] = []
    seen_xrefs: set[int] = set()

    for page_index, page in enumerate(doc):
        if len(figures) >= MAX_FIGURES:
            break
        for img in page.get_images(full=True):
            if len(figures) >= MAX_FIGURES:
                break
            xref = img[0]
            if xref in seen_xrefs:
                continue
            try:
                base = doc.extract_image(xref)
            except Exception:
                continue
            width, height = base.get("width", 0), base.get("height", 0)
            if width < MIN_FIGURE_DIM or height < MIN_FIGURE_DIM:
                continue
            ratio = width / max(height, 1)
            if ratio > 12 or ratio < 1 / 12:
                continue
            seen_xrefs.add(xref)
            ext = base.get("ext", "png")
            mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
            b64 = base64.b64encode(base["image"]).decode("ascii")
            figures.append(
                Figure(
                    index=len(figures) + 1,
                    page=page_index + 1,
                    mime=mime,
                    data_base64=b64,
                )
            )

    page_count = doc.page_count
    doc.close()
    return full_text, figures, page_count, truncated, rag_text


# --------------------------------------------------------------------------
# LLM helpers
# --------------------------------------------------------------------------

def make_client(base_url: str, api_key: str) -> AsyncOpenAI:
    # Generous timeouts: local/small-model servers (Ollama, llama.cpp, LM Studio)
    # can be slow to even start responding under load, and slow to generate.
    timeout = httpx.Timeout(600.0, connect=30.0)
    return AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout)


async def describe_figure(client: AsyncOpenAI, model: str, fig: Figure) -> str:
    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=180,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FIGURE_DESC_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{fig.mime};base64,{fig.data_base64}"},
                        },
                    ],
                }
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Figure description unavailable: {e})"


def render_report(raw_markdown: str, figures: list[Figure]) -> str:
    fig_by_index = {f.index: f for f in figures}

    def replace(match: "re.Match") -> str:
        n = int(match.group(1))
        f = fig_by_index.get(n)
        if not f:
            return ""
        return f"\n![Figure {n}](data:{f.mime};base64,{f.data_base64})\n*Figure {n} — page {f.page}*\n"

    return re.sub(r"\{\{figure:(\d+)\}\}", replace, raw_markdown)


def resolve_provider_config(
    provider: str,
    groq_api_key: Optional[str],
    groq_text_model: Optional[str],
    groq_vision_model: Optional[str] = None,
    custom_base_url: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    custom_text_model: Optional[str] = None,
    custom_vision_model: Optional[str] = None,
) -> tuple[str, str, str, Optional[str]]:
    """Returns (base_url, api_key, text_model, vision_model)."""
    if provider == "custom":
        if not custom_base_url or not custom_text_model:
            raise HTTPException(400, "Custom provider requires base URL and text model name.")
        base_url = custom_base_url.rstrip("/")
        api_key = custom_api_key or os.getenv("CUSTOM_API_KEY", "")
        text_model = custom_text_model
        vision_model = custom_vision_model or os.getenv("CUSTOM_VISION_MODEL") or None
    else:
        base_url = GROQ_BASE_URL
        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise HTTPException(400, "Missing Groq API key. Set it in Settings or in backend/.env")
        text_model = groq_text_model or os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
        vision_model = groq_vision_model or os.getenv(
            "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        )
    return base_url, api_key, text_model, vision_model


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    provider: str = Form("groq"),
    extract_images: bool = Form(True),
    groq_api_key: Optional[str] = Form(None),
    groq_text_model: Optional[str] = Form(None),
    groq_vision_model: Optional[str] = Form(None),
    custom_base_url: Optional[str] = Form(None),
    custom_api_key: Optional[str] = Form(None),
    custom_text_model: Optional[str] = Form(None),
    custom_vision_model: Optional[str] = Form(None),
):
    """
    Streams the whole analysis as Server-Sent Events so you can watch each
    agent write its section live (this matters a lot if `text_model` is a
    small/local model — you actually see tokens land instead of staring at
    a spinner). Event types:
      status         -> {"message": "..."}                         (planning/research/assembly phases)
      section_start  -> {"key": "A"}                                (one of the 6 section agents began)
      section_delta  -> {"key": "A", "delta": "..."}                (a token/chunk from that agent)
      section_done   -> {"key": "A"}                                (that agent finished)
      research_done  -> {"queries": [...], "related_papers_found": N, "used_mcp": bool}
      final          -> {"report_markdown", "figures", "meta", "session_id"}  (fully assembled + rendered)
      error          -> {"message": "..."}
    Every event is one `data: <json>\\n\\n` frame; the stream ends with `data: [DONE]\\n\\n`.
    Each of the 6 sections is still its own independent LLM call with its own
    context window (not one giant prompt), so it can't blow past a small model's
    context limit — they just happen to run concurrently and stream live.
    """
    if not file and not url:
        raise HTTPException(400, "Provide either a PDF file or a paper URL.")

    if file:
        pdf_bytes = await file.read()
        source = file.filename or "uploaded file"
    else:
        pdf_bytes = await resolve_pdf_bytes_from_url(url)
        source = url

    if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
        raise HTTPException(400, "The provided file/URL did not resolve to a valid PDF.")

    try:
        paper_text, figures, page_count, truncated, rag_text = extract_text_and_figures(pdf_bytes)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse PDF: {e}")

    if not extract_images:
        figures = []
    if not paper_text:
        raise HTTPException(400, "No extractable text found (this may be a scanned/image-only PDF).")

    base_url, api_key, text_model, vision_model = resolve_provider_config(
        provider, groq_api_key, groq_text_model, groq_vision_model,
        custom_base_url, custom_api_key, custom_text_model, custom_vision_model,
    )
    client = make_client(base_url, api_key)

    async def event_gen():
        import asyncio
        import json as _json

        def sse(obj: dict) -> str:
            return f"data: {_json.dumps(obj)}\n\n"

        try:
            if figures and vision_model:
                yield sse({"type": "status", "message": f"Reading {len(figures)} figure(s)…"})
                descriptions = await asyncio.gather(
                    *[describe_figure(client, vision_model, f) for f in figures], return_exceptions=True
                )
                for f, d in zip(figures, descriptions):
                    f.description = d if not isinstance(d, Exception) else None
                yield sse({"type": "status", "message": "Figures described."})

            # Local/custom model servers (Ollama, llama.cpp, LM Studio, etc.) often
            # can't handle several simultaneous generations and will drop connections
            # under load — default to a lower concurrency for them. Groq's hosted API
            # handles the full 6-way parallelism fine. Override with SECTION_CONCURRENCY.
            default_concurrency = 2 if provider == "custom" else 6
            max_concurrency = int(os.getenv("SECTION_CONCURRENCY", str(default_concurrency)))

            ctx = AgentContext(
                client=client,
                text_model=text_model,
                paper_text=paper_text,
                truncated=truncated,
                figures=figures,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                max_concurrency=max_concurrency,
            )

            queue: "asyncio.Queue" = asyncio.Queue()

            async def emit(event: dict):
                await queue.put(event)

            result_holder: dict = {}

            async def run_and_close():
                try:
                    report_markdown, findings = await run_pipeline_streaming(ctx, emit)
                    result_holder["report_markdown"] = report_markdown
                    result_holder["findings"] = findings
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    await queue.put(None)  # sentinel

            task = asyncio.create_task(run_and_close())

            while True:
                event = await queue.get()
                if event is None:
                    break
                yield sse(event)

            await task

            if "error" in result_holder:
                yield sse({"type": "error", "message": f"Agent pipeline failed: {result_holder['error']}"})
            else:
                report_markdown = render_report(result_holder["report_markdown"], figures)
                findings = result_holder["findings"]
                session_id = rag.create_session(rag_text, title=source)
                meta = AnalyzeMeta(
                    source=source,
                    pages=page_count,
                    truncated=truncated,
                    figures_found=len(figures),
                    figures_described=sum(1 for f in figures if f.description),
                    text_model=text_model,
                    vision_model=vision_model,
                    research_queries=findings.get("queries", []),
                    related_papers_found=len(findings.get("related_papers", [])),
                    used_mcp=findings.get("used_mcp", False),
                )
                yield sse({
                    "type": "final",
                    "report_markdown": report_markdown,
                    "figures": [f.model_dump() for f in figures],
                    "meta": meta.model_dump(),
                    "session_id": session_id,
                })
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not rag.session_exists(req.session_id):
        raise HTTPException(404, "This paper session has expired or wasn't found. Please re-analyze the paper.")

    base_url, api_key, text_model, _ = resolve_provider_config(
        req.provider, req.groq_api_key, req.groq_text_model, None,
        req.custom_base_url, req.custom_api_key, req.custom_text_model, None,
    )
    client = make_client(base_url, api_key)

    messages = rag.build_chat_messages(
        req.session_id, req.question, [m.model_dump() for m in req.history]
    )
    try:
        resp = await client.chat.completions.create(
            model=text_model, temperature=0.2, max_tokens=800, messages=messages
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(502, f"LLM request failed: {e}")

    return ChatResponse(answer=answer)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Same RAG chat as /api/chat, but streams the answer token-by-token as
    Server-Sent Events: `data: {"delta": "..."}` per chunk, ending with
    `data: [DONE]`. Errors mid-stream are sent as `data: {"error": "..."}`
    (the connection stays 200 OK since headers are already flushed by then)."""
    if not rag.session_exists(req.session_id):
        raise HTTPException(404, "This paper session has expired or wasn't found. Please re-analyze the paper.")

    base_url, api_key, text_model, _ = resolve_provider_config(
        req.provider, req.groq_api_key, req.groq_text_model, None,
        req.custom_base_url, req.custom_api_key, req.custom_text_model, None,
    )
    client = make_client(base_url, api_key)
    messages = rag.build_chat_messages(
        req.session_id, req.question, [m.model_dump() for m in req.history]
    )

    async def event_gen():
        import json as _json
        try:
            stream = await client.chat.completions.create(
                model=text_model, temperature=0.2, max_tokens=800, messages=messages, stream=True
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta else None
                if delta:
                    yield f"data: {_json.dumps({'delta': delta})}\n\n"
        except Exception as e:
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)