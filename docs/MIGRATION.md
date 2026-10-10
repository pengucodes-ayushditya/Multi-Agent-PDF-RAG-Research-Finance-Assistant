# Consolidation notes

The consolidated implementation replaces the independent executable prototypes with a single Python package and LangGraph workflow. Original folders remain available separately.

| Original | Behavior retained | Changes |
|---|---|---|
| `VDB/pdf_assistant.py` | Ollama chat, local embeddings, pgvector PDF retrieval | Configurable local PDFs, explicit ingestion, page references, safe configuration, content-based document IDs |
| `VDB/pdf_assistant_beta.py` | PDF question answering | Replaced incompatible legacy Assistant/PgVector2 APIs and the Groq SDK/client mismatch. Database-backed session persistence from this prototype is not carried into the CLI; chat context is in memory. |
| `VDB/financial_agent.py` | Finance and web tools, source-oriented answers | Replaced PhiData agent/team APIs with a LangGraph specialist. Removed the hardcoded NVDA query that ran at import. Added quote/fundamentals/recommendations/news categories. |
| `LangGraph/chatbotmultipletools.ipynb` | Arxiv, Wikipedia, and Tavily search in a tool loop | Packaged as a research specialist, with DuckDuckGo fallback, execution limits, and structured evidence. Duplicate notebook graph definitions and saved outputs are excluded. |

The original PDF prototypes used a fixed public recipe PDF. The consolidated application indexes a user-selected local file; no external PDF is downloaded or indexed on import. The original finance team was the implemented multi-agent component; the research notebook and PDF scripts each used one tool-enabled model. The new coordinator connects all three domains through a shared source/evidence format.

The lecture-video RAG prototype belongs to KnowledgeOS and is not copied into this repository. This consolidation includes the finance functionality previously listed separately in the portfolio plan.

Credentials from original `.env` files are not copied. Connection details are supplied through environment configuration. No original database contents, screenshots, notebooks with outputs, or PDFs are included in the publication package.
