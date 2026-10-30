"""
research_tools.py
------------------
Plain async functions the Research Agent calls to look things up on the
open web and in literature databases. No agent framework needed — each
function is just an httpx call + a small parser. Every function fails soft
(returns an empty/zero result instead of raising) so one flaky API never
kills the whole pipeline.

Sources:
  - web_search()            -> Tavily API if TAVILY_API_KEY is set,
                                else DuckDuckGo HTML fallback (no key needed)
  - semantic_scholar_search()-> api.semanticscholar.org (no key needed,
                                 free tier) — gives an actual "how many
                                 papers match this" count + top hits
  - arxiv_search()           -> export.arxiv.org Atom API (no key needed)
                                 — another independent "how many papers"
                                 signal plus recent preprints
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (compatible; PaperExplainerBot/1.0; research-agent)"}
TIMEOUT = httpx.Timeout(20.0, connect=10.0)


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Returns [{title, url, snippet}, ...]. Tries Tavily first (better
    quality, needs TAVILY_API_KEY), falls back to scraping DuckDuckGo's
    no-JS HTML endpoint (works with zero configuration)."""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                )
                r.raise_for_status()
                data = r.json()
                return [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": (item.get("content", "") or "")[:400],
                    }
                    for item in data.get("results", [])[:max_results]
                ]
        except Exception:
            pass  # fall through to DuckDuckGo

    try:
        async with httpx.AsyncClient(headers=UA, timeout=TIMEOUT, follow_redirects=True) as client:
            r = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            results = []
            for res in soup.select("div.result")[:max_results]:
                a = res.select_one("a.result__a")
                snippet_el = res.select_one("a.result__snippet") or res.select_one(".result__snippet")
                if not a:
                    continue
                results.append(
                    {
                        "title": a.get_text(strip=True),
                        "url": a.get("href", ""),
                        "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    }
                )
            return results
    except Exception:
        return []


async def semantic_scholar_search(query: str, limit: int = 5) -> dict:
    """Returns {"total": int, "papers": [{title, year, url, citations}, ...]}.
    `total` is Semantic Scholar's own match count for the query, which is
    the closest thing to an authoritative "how many papers like this exist"
    number we can get without an LLM guessing."""
    try:
        async with httpx.AsyncClient(headers=UA, timeout=TIMEOUT) as client:
            r = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,year,url,citationCount,abstract",
                },
            )
            if r.status_code != 200:
                return {"total": 0, "papers": []}
            data = r.json()
            papers = [
                {
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "url": p.get("url", ""),
                    "citations": p.get("citationCount", 0),
                }
                for p in data.get("data", []) or []
            ]
            return {"total": data.get("total", len(papers)), "papers": papers}
    except Exception:
        return {"total": 0, "papers": []}


async def arxiv_search(query: str, max_results: int = 5) -> dict:
    """Returns {"total": int, "papers": [{title, url, published}, ...]}
    using arXiv's Atom search API. `total` comes from opensearch:totalResults."""
    try:
        async with httpx.AsyncClient(headers=UA, timeout=TIMEOUT) as client:
            r = await client.get(
                "http://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": max_results,
                },
            )
            r.raise_for_status()
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "os": "http://a9.com/-/spec/opensearch/1.1/",
            }
            root = ET.fromstring(r.text)
            total_el = root.find("os:totalResults", ns)
            total = int(total_el.text) if total_el is not None and total_el.text else 0
            papers = []
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                id_el = entry.find("atom:id", ns)
                pub_el = entry.find("atom:published", ns)
                papers.append(
                    {
                        "title": re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else "",
                        "url": id_el.text.strip() if id_el is not None and id_el.text else "",
                        "published": pub_el.text[:10] if pub_el is not None and pub_el.text else "",
                    }
                )
            return {"total": total, "papers": papers}
    except Exception:
        return {"total": 0, "papers": []}