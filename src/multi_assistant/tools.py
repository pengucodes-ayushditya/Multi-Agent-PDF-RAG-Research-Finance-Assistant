"""Small, bounded adapters. Network access happens only when a tool is invoked."""

import json
import re
from datetime import UTC, datetime
from typing import Literal

import httpx
from defusedxml import ElementTree
from langchain_core.tools import tool

from .config import Settings
from .documents import DocumentStore
from .evidence import Evidence, Source


class ResearchService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def web(self, query: str) -> Evidence:
        if self.settings.tavily_api_key:
            response = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.settings.tavily_api_key.get_secret_value(),
                    "query": query,
                    "max_results": 5,
                    "include_answer": False,
                },
                timeout=self.settings.request_timeout,
            )
            response.raise_for_status()
            rows = response.json().get("results", [])
            results = [(r.get("title", "Web result"), r["url"], r.get("content", "")) for r in rows]
        else:
            from ddgs import DDGS

            rows = DDGS(timeout=int(self.settings.request_timeout)).text(query, max_results=5)
            results = [(r.get("title", "Web result"), r["href"], r.get("body", "")) for r in rows]
        sources, excerpts = [], []
        for title, url, content in results:
            source = Source.create(title, url)
            sources.append(source)
            excerpts.append(f"[{source.id}] {title}: {content[:1800]}")
        return Evidence(data="\n\n".join(excerpts) or "No web results found.", sources=sources)

    def arxiv(self, query: str) -> Evidence:
        response = httpx.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "max_results": 3,
                "sortBy": "relevance",
            },
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        sources, excerpts = [], []
        for entry in root.findall("a:entry", ns):
            title = " ".join(entry.findtext("a:title", "", ns).split())
            url = entry.findtext("a:id", "", ns)
            summary = " ".join(entry.findtext("a:summary", "", ns).split())
            published = entry.findtext("a:published", "", ns)
            source = Source.create(title, url)
            sources.append(source)
            excerpts.append(f"[{source.id}] {title} ({published}): {summary[:2500]}")
        return Evidence(data="\n\n".join(excerpts) or "No Arxiv results found.", sources=sources)

    def wikipedia(self, query: str) -> Evidence:
        response = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 3,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "exchars": 1800,
                "inprop": "url",
                "format": "json",
            },
            headers={
                "User-Agent": "MultiAgentPDFRAGResearchFinanceAssistant/0.1 (educational CLI)"
            },
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        sources, excerpts = [], []
        for page in pages.values():
            source = Source.create(page["title"], page["fullurl"])
            sources.append(source)
            excerpts.append(f"[{source.id}] {page['title']}: {page.get('extract', '')}")
        return Evidence(
            data="\n\n".join(excerpts) or "No Wikipedia results found.", sources=sources
        )


class FinanceService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def lookup(self, symbol: str, category: str) -> Evidence:
        import yfinance as yf

        symbol = symbol.upper().strip()
        if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=-]{0,19}", symbol):
            raise ValueError("Invalid ticker symbol")
        ticker = yf.Ticker(symbol)
        if category in ("quote", "fundamentals"):
            info = ticker.get_info()
            fields = [
                "symbol",
                "shortName",
                "currency",
                "exchange",
                "regularMarketPrice",
                "regularMarketTime",
                "marketCap",
            ]
            if category == "fundamentals":
                fields += [
                    "sector",
                    "industry",
                    "trailingPE",
                    "forwardPE",
                    "totalRevenue",
                    "revenueGrowth",
                    "profitMargins",
                    "totalDebt",
                    "totalCash",
                ]
            data = {field: info.get(field) for field in fields}
        elif category == "recommendations":
            frame = ticker.get_recommendations_summary()
            data = [] if frame is None else json.loads(frame.head(5).to_json(orient="records"))
        elif category == "news":
            items = ticker.get_news(count=5)
            data = []
            sources = []
            for item in items:
                content = item.get("content", item)
                url = (content.get("canonicalUrl") or {}).get("url") or content.get("link")
                title = content.get("title", "Company news")
                if url:
                    source = Source.create(title, url)
                    sources.append(source)
                    data.append(
                        {
                            "source_id": source.id,
                            "title": title,
                            "published": content.get("pubDate"),
                            "summary": content.get("summary", "")[:1500],
                        }
                    )
            return Evidence(
                data=json.dumps(data, default=str)
                if data
                else f"No company news returned for {symbol}.",
                sources=sources,
            )
        else:
            raise ValueError("Choose quote, fundamentals, recommendations, or news")
        source = Source.create(
            f"Yahoo Finance {symbol}: {category}", f"https://finance.yahoo.com/quote/{symbol}/"
        )
        return Evidence(
            data=json.dumps(
                {
                    "source_id": source.id,
                    "symbol": symbol,
                    "category": category,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "data": data,
                },
                default=str,
            ),
            sources=[source],
        )


def build_tools(settings: Settings, documents=None, research=None, finance=None):
    documents = documents or DocumentStore(settings)
    research = research or ResearchService(settings)
    finance = finance or FinanceService(settings)

    @tool
    def search_pdfs(query: str) -> str:
        """Retrieve indexed PDF passages with page references, using a focused search query."""
        return documents.search(query, 5).model_dump_json()

    @tool
    def search_web(query: str) -> str:
        """Search the web for current information and source URLs."""
        return research.web(query).model_dump_json()

    @tool
    def search_arxiv(query: str) -> str:
        """Search Arxiv paper abstracts. Supports Arxiv query syntax such as all:retrieval."""
        return research.arxiv(query).model_dump_json()

    @tool
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia for background information, with source URLs."""
        return research.wikipedia(query).model_dump_json()

    @tool
    def finance_lookup(
        symbol: str,
        category: Literal["quote", "fundamentals", "recommendations", "news"] = "quote",
    ) -> str:
        """Retrieve Yahoo Finance data for one ticker. Category: quote, fundamentals,
        recommendations, or news. Prices may be delayed; missing fields are unknown."""
        return finance.lookup(symbol, category).model_dump_json()

    return {
        "documents": [search_pdfs] if settings.database_url else [],
        "research": [search_web, search_arxiv, search_wikipedia],
        "finance": [finance_lookup, search_web],
    }
