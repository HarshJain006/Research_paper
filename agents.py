"""
agents.py
---------
A small, dependency-free agent orchestrator. No LangChain — an "agent"
here is just an async function that (a) has a job, (b) gets an
AsyncOpenAI client + shared context, and (c) returns a chunk of the
final report. The orchestrator (`run_pipeline`) wires them together and
runs the independent ones concurrently with asyncio.gather.

Pipeline:
  1. Planner Agent      -> title, search queries, checkable claims
  2. Research Agent      -> runs those queries against the web / Semantic
                             Scholar / arXiv (and MCP if configured) in
                             parallel, and de-dupes the results
  3. Section Agents (x6) -> run concurrently, each writes 2-3 of the
                             report's sections; the two research-aware
                             ones (Related Work, Research Ideas) get the
                             Research Agent's findings injected
  4. Assembler            -> concatenates sections in order + appends a
                             programmatic "Sources Consulted" section
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from openai import AsyncOpenAI

from research_tools import arxiv_search, semantic_scholar_search, web_search
from mcp_client import mcp_configured, mcp_search


@dataclass
class AgentContext:
    client: AsyncOpenAI
    text_model: str
    paper_text: str
    truncated: bool
    figures: list
    max_output_tokens: int = 1400
    max_concurrency: int = 6  # lower this for local/small model servers that can't handle parallel generations
    progress_cb: Optional[Callable[[dict], "asyncio.Future"]] = None  # emit(event_dict) -> None


async def _emit(ctx: AgentContext, event: dict):
    if ctx.progress_cb:
        try:
            await ctx.progress_cb(event)
        except Exception:
            pass


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip()


async def _call(ctx: AgentContext, system: str, user: str, max_tokens: Optional[int] = None, temperature: float = 0.3) -> str:
    resp = await ctx.client.chat.completions.create(
        model=ctx.text_model,
        temperature=temperature,
        max_tokens=max_tokens or ctx.max_output_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------
# 1. Planner Agent
# --------------------------------------------------------------------------

PLANNER_SYSTEM = """Triage agent. Read the paper excerpt. Output ONLY this JSON, nothing else:
{"title": "paper title", "field": "one-line research area", "queries": ["2-4 short search queries for related work / verifying the main claim"], "claims": ["1-3 specific checkable claims, e.g. a named method + its result"]}"""


async def run_planner(ctx: AgentContext) -> dict:
    await _emit(ctx, {"type": "status", "message": "Planning: identifying claims to verify and topics to research…"})
    raw = await _call(ctx, PLANNER_SYSTEM, ctx.paper_text[:12000], max_tokens=500, temperature=0.1)
    try:
        data = json.loads(_strip_fences(raw))
        data.setdefault("title", "Untitled paper")
        data.setdefault("queries", [])
        data.setdefault("claims", [])
        return data
    except Exception:
        return {"title": "Untitled paper", "field": "", "queries": [], "claims": []}


# --------------------------------------------------------------------------
# 2. Research Agent (parallel web + literature lookups)
# --------------------------------------------------------------------------

async def _search_one(query: str) -> dict:
    mcp_result = await mcp_search(query) if mcp_configured() else None
    web_task = web_search(query) if mcp_result is None else _noop_list()
    web_r, s2_r, arx_r = await asyncio.gather(web_task, semantic_scholar_search(query), arxiv_search(query), return_exceptions=True)
    web = mcp_result if mcp_result is not None else (web_r if not isinstance(web_r, Exception) else [])
    s2 = s2_r if not isinstance(s2_r, Exception) else {"total": 0, "papers": []}
    arx = arx_r if not isinstance(arx_r, Exception) else {"total": 0, "papers": []}
    return {"query": query, "web": web, "semantic_scholar": s2, "arxiv": arx}


async def _noop_list():
    return []


async def run_research(ctx: AgentContext, plan: dict) -> dict:
    await _emit(ctx, {"type": "status", "message": "Researching: searching the web, Semantic Scholar, and arXiv…"})
    queries = [q for q in (plan.get("queries") or []) if q][:5]
    if not queries and plan.get("title"):
        queries = [plan["title"]]
    if not queries:
        return {"queries": [], "web": [], "related_papers": [], "counts": {}, "used_mcp": mcp_configured()}

    per_query = await asyncio.gather(*[_search_one(q) for q in queries], return_exceptions=True)

    web_hits, related_papers, counts = [], [], {}
    for item in per_query:
        if isinstance(item, Exception):
            continue
        web_hits.extend(item["web"] or [])
        related_papers.extend((item["semantic_scholar"] or {}).get("papers", []) or [])
        related_papers.extend(
            [{"title": p["title"], "url": p["url"], "year": p.get("published", "")[:4]} for p in (item["arxiv"] or {}).get("papers", []) or []]
        )
        counts[item["query"]] = {
            "semantic_scholar": (item["semantic_scholar"] or {}).get("total", 0),
            "arxiv": (item["arxiv"] or {}).get("total", 0),
        }

    seen, dedup = set(), []
    for p in related_papers:
        key = (p.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            dedup.append(p)

    return {
        "queries": queries,
        "web": web_hits[:15],
        "related_papers": dedup[:12],
        "counts": counts,
        "used_mcp": mcp_configured(),
    }


def _findings_as_text(findings: dict) -> str:
    if not findings.get("queries"):
        return "(No external research was performed.)"
    lines = ["RESEARCH FINDINGS (from live web + literature search — treat as ground truth for novelty/counts):"]
    for q, c in findings.get("counts", {}).items():
        lines.append(f"- Query \"{q}\": ~{c.get('semantic_scholar', 0)} Semantic Scholar matches, ~{c.get('arxiv', 0)} arXiv matches")
    if findings.get("related_papers"):
        lines.append("\nTOP RELATED PAPERS FOUND:")
        for p in findings["related_papers"][:10]:
            year = p.get("year") or "?"
            lines.append(f"- {p.get('title','(untitled)')} ({year}) — {p.get('url','')}")
    if findings.get("web"):
        lines.append("\nWEB SEARCH SNIPPETS:")
        for w in findings["web"][:8]:
            snippet = (w.get("snippet") or "").replace("\n", " ")[:220]
            lines.append(f"- {w.get('title','')}: {snippet} ({w.get('url','')})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 3. Section Writer Agents (run concurrently)
# --------------------------------------------------------------------------

BASE_RULES = """Rules:
- Keep "## N. Heading" lines exactly as given.
- Markdown only: **bold** for key terms, "- " for bullets. Never use "*" alone.
- Be brief. Short bullets over paragraphs. Cut anything not essential.
- Math: $inline$ or $$display$$, real LaTeX. Never use $ for anything else.
- To show a figure, write {{figure:N}} alone on a line (only numbers listed below), then a 1-line caption.
- At most one ```mermaid``` diagram, only if it truly helps.
- Mark guesses as (inference)."""

LENGTH_CAP = "Stay under {n} words total. Cut before you pad."


def _figure_list_text(figures: list) -> str:
    if not figures:
        return "(No figures were extracted from this paper.)"
    lines = []
    for f in figures:
        desc = f.description or "(no description available)"
        lines.append(f"- Figure {f.index} (page {f.page}): {desc}")
    return "\n".join(lines)


def _section_user(ctx: AgentContext, plan: dict, extra_user_block: str = "") -> str:
    """Every section call is fully independent (own LLM call, own context window),
    but all of them get this same header so a small/local model doesn't drift on
    the paper's identity or framing across calls — cheap consistency without making
    the calls depend on each other's output."""
    anchor = f"PAPER TITLE (from triage): {plan.get('title', 'Unknown')}\nRESEARCH FIELD: {plan.get('field', 'unspecified')}\n\n"
    return (
        f"{anchor}"
        f"PAPER TEXT{' (truncated for length)' if ctx.truncated else ''}:\n\n{ctx.paper_text}\n\n"
        f"EXTRACTED FIGURES:\n{_figure_list_text(ctx.figures)}\n\n{extra_user_block}"
    )


async def _stream_call(ctx: AgentContext, key: str, system: str, user: str, emit, semaphore: "asyncio.Semaphore") -> str:
    """Streams one section's own, independent LLM call token-by-token via `emit`,
    so the caller (an SSE endpoint) can show it live — this is what lets you watch
    a small local model (e.g. Gemma 3) actually generate instead of waiting on
    one big blocking call.

    Two protections for local/small model servers, which often can't handle
    several simultaneous generations:
      - `semaphore` caps how many sections run at once (see AgentContext.max_concurrency).
      - If a connection error happens before any token for this section has been
        received, it's silently retried (a busy local server dropping a connection
        under load is transient); once a token has actually streamed, we stop
        retrying that section (retrying would duplicate output already shown)
        and just append a short failure note instead.
    """
    await emit({"type": "section_start", "key": key})
    chunks: list[str] = []
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        got_any = False
        try:
            async with semaphore:
                stream = await ctx.client.chat.completions.create(
                    model=ctx.text_model,
                    temperature=0.3,
                    max_tokens=ctx.max_output_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    stream=True,
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content if chunk.choices[0].delta else None
                    if delta:
                        got_any = True
                        chunks.append(delta)
                        await emit({"type": "section_delta", "key": key, "delta": delta})
            break  # completed this attempt without an exception
        except Exception as e:
            if not got_any and attempt < max_attempts:
                # nothing was shown to the user yet for this section — safe to
                # quietly retry, likely a local server momentarily overloaded
                await emit({"type": "status", "message": f"{key}: connection hiccup, retrying ({attempt}/{max_attempts})…"})
                await asyncio.sleep(min(1.5 * attempt, 5))
                continue
            err = f"\n\n_(this section failed mid-generation: {e})_"
            chunks.append(err)
            await emit({"type": "section_delta", "key": key, "delta": err})
            break

    text = "".join(chunks).strip()
    await emit({"type": "section_done", "key": key})
    return text


SYS_A = f"""Expert AI research scientist. Write EXACTLY these 3 sections, in order:

## 1. Executive Summary
4 one-line bullets: Problem / Solution / Contribution / Why it matters.

## 2. Technical Explanation
Method, architecture, training, results. Condensed bullets or a short paragraph.

## 3. Layman's Explanation
2-4 plain sentences, one analogy, no jargon.

{LENGTH_CAP.format(n=280)}
{BASE_RULES}"""

SYS_B = f"""Expert AI research scientist. Write EXACTLY these 3 sections, in order:

## 4. Innovation
Max 3 bullets: what's new, why prior work fell short, why this is better.

## 5. Author's Thinking Process
3-4 bullets inferring motivation/trade-offs. Mark as (inference).

## 6. Mathematics
What the key equations compute and why, minimal derivation. If none, say so in one line.

{LENGTH_CAP.format(n=280)}
{BASE_RULES}"""

SYS_C = f"""Expert AI research scientist. Write EXACTLY these 3 sections, in order:

## 7. Experimental Results
Datasets, metrics, key numbers. Bullets only, no narration.

## 8. Strengths & Limitations
3-4 bullets total, combined.

## 9. Practical Applications
2-3 bullets, most plausible uses only.

{LENGTH_CAP.format(n=220)}
{BASE_RULES}"""

SYS_TAKEAWAYS = f"""Expert AI research scientist. Write EXACTLY:

## 12. Key Takeaways
5-7 one-line bullets covering the whole paper.

{LENGTH_CAP.format(n=120)}
{BASE_RULES}"""

SYS_RELATED = f"""Literature-review agent. You're given live web/Semantic Scholar/arXiv results —
treat their counts as ground truth, never invent numbers. Write EXACTLY:

## 10. Related Work & Novelty Check
- 3-4 bullets comparing this paper to what search found (name real papers found only).
- One **"Novelty signal"** line: crowded or unexplored, based on the counts.
- One line: do the paper's claims look consistent with what search found (or "no relevant results").

{LENGTH_CAP.format(n=180)}
{BASE_RULES}"""

SYS_IDEAS = f"""Research-ideation agent. Goal: help the reader plan their OWN follow-up paper.
You're given live web/Semantic Scholar/arXiv results — use the counts as a real novelty
signal, not a guess. Write EXACTLY:

## 11. Research Paper Ideas — Build On This Work
3-5 numbered ideas. Each idea = 4 one-line sub-bullets:
- **Idea**: one sentence.
- **Why it's open**: the specific gap.
- **Novelty signal**: actual counts, e.g. "~12 S2 hits, ~4 arXiv — underexplored".
- **First experiment**: smallest test, one sentence.

Most important section — specific, not generic.
{LENGTH_CAP.format(n=320)}
{BASE_RULES}"""

# (key, human-readable title, system prompt, needs_research_findings)
SECTION_SPECS = [
    ("A", "Executive Summary · Technical Explanation · Layman's Explanation", SYS_A, False),
    ("B", "Innovation · Author's Thinking · Mathematics", SYS_B, False),
    ("C", "Experimental Results · Strengths & Limitations · Applications", SYS_C, False),
    ("takeaways", "Key Takeaways", SYS_TAKEAWAYS, False),
    ("related", "Related Work & Novelty Check", SYS_RELATED, True),
    ("ideas", "Research Paper Ideas", SYS_IDEAS, True),
]


# --------------------------------------------------------------------------
# 4. Assembler
# --------------------------------------------------------------------------

def _sources_section(findings: dict) -> str:
    lines = ["## 13. Sources Consulted"]
    if findings.get("used_mcp"):
        lines.append("_Retrieved live via a connected MCP research server, plus Semantic Scholar and arXiv._")
    if not findings.get("queries"):
        lines.append("No external research was performed for this report.")
        return "\n".join(lines)
    lines.append("**Search queries used:** " + ", ".join(f"`{q}`" for q in findings["queries"]))
    if findings.get("related_papers"):
        lines.append("\n**Papers found:**")
        for p in findings["related_papers"]:
            year = f" ({p['year']})" if p.get("year") else ""
            url = p.get("url", "")
            lines.append(f"- [{p.get('title','(untitled)')}]({url}){year}" if url else f"- {p.get('title','(untitled)')}{year}")
    if findings.get("web"):
        lines.append("\n**Web sources:**")
        for w in findings["web"][:8]:
            url = w.get("url", "")
            title = w.get("title") or url
            lines.append(f"- [{title}]({url})" if url else f"- {title}")
    return "\n".join(lines)


async def run_pipeline_streaming(ctx: AgentContext, emit) -> tuple[str, dict]:
    """Streaming version of the pipeline. `emit(event_dict)` is called for every
    status update and every token of every section, as they happen — each section
    is still its own independent LLM call (own context window, no risk of blowing
    past a small model's limit), they just run concurrently and each one's tokens
    are forwarded live instead of being held back until the whole pipeline finishes.
    Returns (report_markdown, findings) once everything is done, same as before.
    """
    plan = await run_planner(ctx)
    await emit({"type": "status", "message": f"Plan ready — \"{plan.get('title','Untitled')}\" ({plan.get('field','')})."})

    semaphore = asyncio.Semaphore(max(1, ctx.max_concurrency))

    # Research runs in the background; sections that don't need it can start immediately.
    research_task = asyncio.create_task(run_research(ctx, plan))

    independent_keys = [k for k, _, _, needs_findings in SECTION_SPECS if not needs_findings]
    tasks: dict[str, "asyncio.Task[str]"] = {}
    for key, _title, system, _needs in SECTION_SPECS:
        if key in independent_keys:
            user = _section_user(ctx, plan)
            tasks[key] = asyncio.create_task(_stream_call(ctx, key, system, user, emit, semaphore))

    findings = await research_task
    await emit({
        "type": "research_done",
        "queries": findings.get("queries", []),
        "related_papers_found": len(findings.get("related_papers", [])),
        "used_mcp": findings.get("used_mcp", False),
    })

    findings_block = _findings_as_text(findings)
    claims_block = (
        "CHECKABLE CLAIMS IDENTIFIED:\n" + "\n".join(f"- {c}" for c in plan.get("claims", []))
        if plan.get("claims") else ""
    )
    for key, _title, system, needs_findings in SECTION_SPECS:
        if needs_findings:
            user = _section_user(ctx, plan, f"{findings_block}\n\n{claims_block}")
            tasks[key] = asyncio.create_task(_stream_call(ctx, key, system, user, emit, semaphore))

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    section_texts: dict[str, str] = {}
    for key, res in zip(tasks.keys(), results):
        section_texts[key] = res if not isinstance(res, Exception) else f"## (section unavailable: {res})"

    await emit({"type": "status", "message": "Assembling final report…"})
    ordered = [
        section_texts["A"],
        section_texts["B"],
        section_texts["C"],
        section_texts["related"],
        section_texts["ideas"],
        section_texts["takeaways"],
        _sources_section(findings),
    ]
    report_markdown = "\n\n".join(s.strip() for s in ordered if s and s.strip())
    return report_markdown, findings