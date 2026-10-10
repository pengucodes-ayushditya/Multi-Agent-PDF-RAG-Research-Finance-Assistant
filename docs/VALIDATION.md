# Validation record

Validated locally on October 10, 2026, with Python 3.11.15, uv 0.11.29, LangGraph 1.2.14, Ollama, and PostgreSQL 17 with pgvector. Exact Python package versions are recorded in `uv.lock`.

| Check | Observed result |
|---|---|
| Offline test suite | 26 passed; the opt-in integration test was skipped in the default run |
| Real pgvector integration | 1 passed: extraction, embeddings, insertion, retrieval, document re-ingestion, collection isolation, and dimension mismatch handling |
| Ruff lint and format | Passed |
| Locked dependency installation | Passed |
| Wheel and source distribution | Built successfully; archive contents excluded caches, PDFs, `.env`, and test temporary files |
| Publication-file review | 21 repository files reviewed; no original API keys, private PDFs, datasets, or notebook outputs included |

## Live provider and tool checks

- Ollama `llama3.2` completed a PDF question using `nomic-embed-text` embeddings and pgvector. The response reported the value in an original, generated test PDF with a page reference.
- Automatic coordinator planning selected documents, finance, and research for a combined question. Each specialist made one tool call. The completed run returned five source records and no tool warnings.
- Arxiv, Wikipedia, DuckDuckGo, and Tavily returned search results with source records.
- Yahoo Finance returned quote, fundamental, and analyst-recommendation data. The company-news endpoint returned no articles for the tested ticker; the application now reports that absence explicitly.
- Groq `openai/gpt-oss-20b` completed JSON planning and general chat, and separately completed a PDF specialist tool call and final synthesis with a source reference. Existing local credentials were read only for these checks and were not copied into the repository.

## Problems found and corrected

- The local model supplied a null optional PDF-tool argument and used a citation placeholder as its search query. The PDF tool now has a simpler query-only schema, and the specialist prompt distinguishes search queries from citations. Validation failures return schema feedback so the model can correct its arguments.
- The installed Ollama server rejected JSON-schema grammar for the coordinator plan. JSON mode with Pydantic validation completed the same plan successfully. Invalid plans receive one bounded retry, followed by a clear message recommending explicit agent selection.
- The original Groq model was unavailable. The example configuration was updated to a model returned by the provider's available-models endpoint and tested successfully.
- A repeat research request encountered an Arxiv HTTP failure. The workflow reported the failed tool and absence of research evidence. A subsequent combined run using Wikipedia completed successfully.
- Final synthesis originally received only specialist summaries. It now also receives bounded excerpts of the retrieved tool evidence, alongside source identifiers, so it can check specialist interpretations.

These are functional smoke checks, not accuracy benchmarks. Small local models can select unsuitable tools or misinterpret evidence. Source records remain available for review, and generated claims are not automatically verified for semantic support. Hosted GitHub Actions has been configured but has not run before publication.
