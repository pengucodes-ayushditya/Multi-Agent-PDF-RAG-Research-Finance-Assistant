import pytest
from pydantic import ValidationError

from multi_assistant.cli import main
from multi_assistant.config import Settings
from multi_assistant.evidence import Evidence, Source
from multi_assistant.tools import FinanceService, build_tools


def test_environment_overrides_file_without_leaking_secrets(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CHAT_MODEL=file-model\nDATABASE_URL=postgresql://private-secret\n")
    monkeypatch.setenv("CHAT_MODEL", "environment-model")
    settings = Settings.load(env)
    assert settings.chat_model == "environment-model"
    assert "private-secret" not in repr(settings)


def test_groq_requires_key():
    with pytest.raises(ValidationError):
        Settings(model_provider="groq")


def test_documents_are_disabled_without_database():
    registry = build_tools(Settings())
    assert registry["documents"] == []
    assert {t.name for t in registry["research"]} == {
        "search_web",
        "search_arxiv",
        "search_wikipedia",
    }


def test_tool_adapters_preserve_sources_and_reject_invalid_ticker():
    class Research:
        def web(self, query):
            return Evidence(data=query, sources=[Source.create("Title", "https://example.test")])

    tools = build_tools(Settings(), research=Research())
    output = Evidence.model_validate_json(tools["research"][0].invoke({"query": "question"}))
    assert output.data == "question"
    assert output.sources[0].locator == "https://example.test"
    with pytest.raises(ValueError, match="ticker"):
        FinanceService(Settings()).lookup("bad symbol", "quote")


def test_cli_configuration_error_does_not_expose_secret(tmp_path, capsys):
    path = tmp_path / ".env"
    path.write_text("DATABASE_URL=postgresql://secret\nMAX_AGENT_TURNS=private-token\n")
    assert main(["--env-file", str(path), "doctor"]) == 1
    assert "private-token" not in capsys.readouterr().err


def test_finance_news_preserves_article_source_and_empty_results(monkeypatch):
    import yfinance

    class Ticker:
        def get_news(self, count):
            return [
                {
                    "content": {
                        "title": "Company update",
                        "summary": "An update",
                        "pubDate": "2026-10-10",
                        "canonicalUrl": {"url": "https://example.test/news"},
                    }
                }
            ]

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: Ticker())
    evidence = FinanceService(Settings()).lookup("MSFT", "news")
    assert evidence.sources[0].locator == "https://example.test/news"
    assert evidence.sources[0].id in evidence.data
    monkeypatch.setattr(Ticker, "get_news", lambda self, count: [])
    assert "No company news returned" in FinanceService(Settings()).lookup("MSFT", "news").data
