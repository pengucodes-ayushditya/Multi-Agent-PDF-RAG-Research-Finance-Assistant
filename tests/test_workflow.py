import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from multi_assistant.agents import Specialist, SpecialistResult
from multi_assistant.config import Settings
from multi_assistant.evidence import Evidence, Source
from multi_assistant.graph import Assistant, Plan, Task, findings_text, render_answer


class Planner:
    def __init__(self, plan):
        self.plan = plan

    def invoke(self, messages):
        return self.plan


class Model:
    def __init__(self, responses=None, plan=None):
        self.responses = iter(responses or [AIMessage(content="Combined answer")])
        self.plan = plan or Plan()
        self.messages = []

    def with_structured_output(self, schema, **kwargs):
        return Planner(self.plan)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.messages.append(messages)
        return next(self.responses)


class StubSpecialist:
    def __init__(self, name):
        self.name = name
        self.received = []
        self.source = Source.create(name, "https://example.test/" + name)

    def run(self, query):
        self.received.append(query)
        return SpecialistResult(
            agent=self.name, answer="Evidence from " + self.name, sources=[self.source]
        )


def call(name="lookup", identity="call1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": identity}])


def test_combined_routing_passes_findings_and_deduplicates_sources():
    specialists = {name: StubSpecialist(name) for name in ("documents", "research", "finance")}
    plan = Plan(tasks=[Task(agent=name, query="specific " + name) for name in specialists])
    model = Model(plan=plan)
    result = Assistant(Settings(), model=model, specialists=specialists).ask("Compare evidence")
    assert [r.agent for r in result["results"]] == list(specialists)
    assert "Evidence from documents" in specialists["research"].received[0]
    assert "Evidence from research" in specialists["finance"].received[0]
    assert len(result["sources"]) == 3
    assert "Sources retrieved:" in render_answer(result)


def test_explicit_agents_bypass_planner_and_unavailable_agent_is_clear():
    specialist = StubSpecialist("research")
    assistant = Assistant(Settings(), model=Model(), specialists={"research": specialist})
    result = assistant.ask("Search", agents=["research"])
    assert result["results"][0].agent == "research"
    with pytest.raises(ValueError, match="unavailable"):
        assistant.ask("Read PDF", agents=["documents"])


def test_plan_rejects_repeated_agents():
    with pytest.raises(ValidationError):
        Plan(tasks=[Task(agent="research", query="a"), Task(agent="research", query="b")])


def test_conversation_can_finish_without_tools_and_retains_context():
    model = Model()
    assistant = Assistant(Settings(), model=model, specialists={})
    history = [HumanMessage(content="My subject is retrieval")]
    result = assistant.ask("Explain it simply", history=history)
    assert result["sources"] == []
    assert history[0] in model.messages[0]


def test_fabricated_source_is_flagged():
    model = Model([AIMessage(content="A claim [S-made-up].")])
    result = Assistant(Settings(), model=model, specialists={}).ask("Hi")
    assert "[unverified citation]" in result["answer"]
    assert result["warnings"]


def test_specialist_consumes_tool_result_and_preserves_evidence():
    source = Source.create("Test PDF", "pdf:test", 2)

    @tool
    def lookup() -> str:
        """Retrieve sample evidence."""
        return Evidence(data="Document passage", sources=[source]).model_dump_json()

    model = Model([call(), AIMessage(content="Supported answer")])
    result = Specialist("documents", model, [lookup]).run("Question")
    assert result.sources == [source]
    assert result.tool_calls == 1
    assert result.evidence[0].data == "Document passage"
    assert "Document passage" in model.messages[1][-1].content


def test_tool_failures_hide_secrets_and_budget_forces_final_answer():
    @tool
    def lookup() -> str:
        """Fail while retrieving evidence."""
        raise RuntimeError("password=super-secret")

    model = Model([call(), AIMessage(content="No verified answer available")])
    result = Specialist("research", model, [lookup], max_tool_calls=1).run("Question")
    assert result.tool_calls == 1
    assert any("RuntimeError" in error for error in result.errors)
    assert "super-secret" not in json.dumps(result.model_dump())
    assert "Tool budget reached" in model.messages[-1][-1].content


def test_multiple_calls_in_one_turn_do_not_exceed_budget():
    invoked = []

    @tool
    def lookup() -> str:
        """Retrieve sample evidence."""
        invoked.append(True)
        return Evidence(data="value").model_dump_json()

    message = AIMessage(
        content="", tool_calls=[{"name": "lookup", "args": {}, "id": str(i)} for i in range(3)]
    )
    model = Model([message, AIMessage(content="Done")])
    result = Specialist("research", model, [lookup], max_tool_calls=1).run("Question")
    assert len(invoked) == result.tool_calls == 1
    assert "Tool-call budget exhausted" in result.errors


def test_unknown_tool_does_not_crash_specialist():
    @tool
    def lookup() -> str:
        """Retrieve sample evidence."""
        return Evidence(data="value").model_dump_json()

    model = Model([call("nonexistent"), AIMessage(content="Unavailable")])
    result = Specialist("research", model, [lookup]).run("Question")
    assert "Unknown tool requested" in result.errors


def test_repeating_model_stops_at_turn_limit():
    @tool
    def lookup() -> str:
        """Retrieve sample evidence."""
        return Evidence(data="value").model_dump_json()

    model = Model([call(identity="1"), call(identity="2"), AIMessage(content="Done")])
    result = Specialist("research", model, [lookup], max_turns=2).run("Question")
    assert result.tool_calls == 2
    assert result.answer == "Done"


def test_invalid_tool_arguments_receive_schema_feedback_then_recover():
    source = Source.create("Evidence", "https://example.test")

    @tool
    def lookup(query: str) -> str:
        """Retrieve sample evidence."""
        return Evidence(data=query, sources=[source]).model_dump_json()

    missing = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "1"}])
    corrected = AIMessage(
        content="", tool_calls=[{"name": "lookup", "args": {"query": "corrected"}, "id": "2"}]
    )
    model = Model([missing, corrected, AIMessage(content="Supported answer")])
    result = Specialist("research", model, [lookup]).run("Question")
    assert result.sources == [source]
    assert result.tool_calls == 2
    assert "schema" in model.messages[1][-1].content


def test_findings_are_bounded_valid_json_and_report_omitted_sources():
    result = SpecialistResult(
        agent="research",
        answer="a" * 5000,
        sources=[Source.create("Title", f"https://example.test/{i}") for i in range(20)],
    )
    data = json.loads(findings_text([result]))[0]
    assert len(data["answer"]) == 1500
    assert len(data["sources"]) == 12
    assert data["omitted_sources"] == 8


def test_invalid_planner_output_has_clear_recovery_message():
    from langchain_core.exceptions import OutputParserException

    class InvalidPlanner:
        def invoke(self, messages):
            raise OutputParserException("Invalid JSON")

    class InvalidModel(Model):
        def with_structured_output(self, schema, **kwargs):
            return InvalidPlanner()

    assistant = Assistant(Settings(), model=InvalidModel(), specialists={})
    with pytest.raises(ValueError, match="Use --agents"):
        assistant.ask("Question")


def test_unknown_source_with_page_suffix_is_flagged():
    model = Model([AIMessage(content="Claim [S-invented page 2].")])
    result = Assistant(Settings(), model=model, specialists={}).ask("Hi")
    assert "[unverified citation]" in result["answer"]
