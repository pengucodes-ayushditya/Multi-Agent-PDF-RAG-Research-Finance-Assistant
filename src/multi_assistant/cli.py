import argparse
import json
import sys
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from .config import Settings
from .documents import DocumentStore


def doctor(settings: Settings):
    checks = {}
    try:
        response = httpx.get(settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=5)
        response.raise_for_status()
        names = [m["name"] for m in response.json()["models"]]

        def has_model(model):
            return any(n == model or n == model + ":latest" for n in names)

        checks["ollama_embeddings"] = (
            "ok" if has_model(settings.embedding_model) else "model missing"
        )
        if settings.model_provider == "ollama":
            checks["chat_model"] = "ok" if has_model(settings.chat_model) else "model missing"
    except Exception as exc:
        checks["ollama"] = f"unavailable ({type(exc).__name__})"
    if settings.model_provider == "groq":
        checks["groq"] = "configured (not contacted)"
    if settings.database_url:
        try:
            with DocumentStore(settings).connect() as connection:
                connection.execute("SELECT 1")
            checks["postgresql"] = "ok"
        except Exception as exc:
            checks["postgresql"] = f"unavailable ({type(exc).__name__})"
    else:
        checks["documents"] = "disabled: DATABASE_URL unset"
    checks["web_search"] = "Tavily configured" if settings.tavily_api_key else "DuckDuckGo"
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multi-agent research, PDF, and finance assistant")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Ask a question; coordinator selects specialists")
    ask.add_argument("question")
    ask.add_argument("--agents", help="Comma-separated documents,research,finance; bypass planner")
    ask.add_argument("--json", action="store_true", help="Include specialist results and sources")
    sub.add_parser("chat", help="Interactive chat; context lasts for this process only")
    ingest = sub.add_parser("ingest", help="Index a local PDF into the configured collection")
    ingest.add_argument("pdf", type=Path)
    retrieve = sub.add_parser("retrieve", help="Retrieve PDF passages without a chat model")
    retrieve.add_argument("query")
    sub.add_parser("documents", help="List indexed PDFs in this collection")
    sub.add_parser("doctor", help="Check local services without showing secrets")
    args = parser.parse_args(argv)
    try:
        settings = Settings.load(args.env_file)
    except ValidationError:
        print(
            "Invalid configuration. Check field names, values, and provider credentials "
            "against .env.example.",
            file=sys.stderr,
        )
        return 1
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(settings), indent=2))
        elif args.command == "ingest":
            count = DocumentStore(settings).ingest(args.pdf)
            print(f"Indexed {count} chunks from {args.pdf.name}")
        elif args.command == "retrieve":
            print(DocumentStore(settings).search(args.query).model_dump_json(indent=2))
        elif args.command == "documents":
            print(json.dumps(DocumentStore(settings).inventory(), indent=2))
        else:
            from .graph import Assistant, render_answer

            assistant = Assistant(settings)
            if args.command == "ask":
                agents = [a.strip() for a in args.agents.split(",")] if args.agents else None
                result = assistant.ask(args.question, agents)
                if args.json:
                    print(
                        json.dumps(
                            result, default=lambda obj: obj.model_dump(mode="json"), indent=2
                        )
                    )
                else:
                    print(render_answer(result))
            else:
                history = []
                print("Enter a question, or /quit to exit. Chat history is held in memory.")
                while True:
                    query = input("You> ").strip()
                    if query == "/quit":
                        break
                    if not query:
                        continue
                    result = assistant.ask(query, history=history)
                    print(render_answer(result))
                    history.extend(
                        [HumanMessage(content=query), AIMessage(content=result["answer"])]
                    )
                    history = history[-10:]
        return 0
    except ValidationError:
        print(
            "Model output or tool arguments failed validation. Try --agents to select "
            "specialists explicitly, or choose another chat model.",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        return 0
    except Exception as exc:
        print(
            f"Operation failed ({type(exc).__name__}). Check doctor, provider availability, "
            "and configuration. Credentials and connection strings are hidden.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
