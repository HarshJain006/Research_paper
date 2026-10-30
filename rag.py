"""
rag.py
------
Minimal retrieval-augmented chat over a single analyzed paper. No vector
DB needed — BM25 over word tokens is plenty for one document and keeps
the dependency list tiny (rank_bm25 + numpy only).

Flow:
  1. /api/analyze calls create_session(text) once per paper -> session_id
  2. /api/chat calls retrieve(session_id, question) for the top-k chunks,
     then asks the LLM to answer using ONLY those chunks.

Sessions live in memory and expire after SESSION_TTL_SECONDS of inactivity
(simple background sweep on access) — fine for a single-user/local tool;
swap for Redis if you need multi-instance deployment.
"""

import re
import time
import uuid
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
SESSION_TTL_SECONDS = 6 * 60 * 60  # 6 hours


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - CHUNK_OVERLAP
    return chunks


@dataclass
class Session:
    title: str
    chunks: list[str]
    bm25: BM25Okapi
    last_used: float = field(default_factory=time.time)


_sessions: dict[str, Session] = {}


def create_session(paper_text: str, title: str = "Untitled paper") -> str:
    _sweep()
    chunks = _chunk_text(paper_text) or [paper_text[:CHUNK_SIZE]]
    tokenized = [_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    session_id = uuid.uuid4().hex
    _sessions[session_id] = Session(title=title, chunks=chunks, bm25=bm25)
    return session_id


def session_exists(session_id: str) -> bool:
    return session_id in _sessions


def retrieve(session_id: str, query: str, k: int = 5) -> list[str]:
    session = _sessions.get(session_id)
    if not session:
        return []
    session.last_used = time.time()
    scores = session.bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top = [session.chunks[i] for i in ranked[:k] if scores[i] > 0]
    return top or session.chunks[:k]  # if nothing scores, just return the first chunks


def _sweep():
    now = time.time()
    dead = [sid for sid, s in _sessions.items() if now - s.last_used > SESSION_TTL_SECONDS]
    for sid in dead:
        _sessions.pop(sid, None)


CHAT_SYSTEM_PROMPT = """You are a research-paper Q&A assistant. Answer the user's question using
ONLY the paper excerpts provided below. Be precise and concise — a few sentences or short
bullets, not an essay.
- If the excerpts don't contain the answer, say so plainly instead of guessing.
- Quote specific numbers/terms from the excerpts when relevant.
- Write any math as real LaTeX: inline as $...$, standalone equations as $$...$$. Never use
  $ for anything except math.
- You may use light Markdown (bold, bullets) but keep answers focused — this is a chat, not a report.
"""


def build_chat_messages(session_id: str, question: str, history: list[dict]) -> list[dict]:
    context_chunks = retrieve(session_id, question, k=5)
    context_block = "\n\n---\n\n".join(context_chunks)
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    # keep last few turns of prior conversation for follow-ups
    for turn in history[-6:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": f"PAPER EXCERPTS (most relevant to your question):\n\n{context_block}\n\nQUESTION: {question}",
        }
    )
    return messages