import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Source(BaseModel):
    id: str
    title: str
    locator: str
    retrieved_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    page: int | None = None

    @classmethod
    def create(cls, title: str, locator: str, page: int | None = None):
        identity = hashlib.sha256(f"{locator}:{page}".encode()).hexdigest()[:12]
        return cls(id=f"S-{identity}", title=title, locator=locator, page=page)


class Evidence(BaseModel):
    data: str
    sources: list[Source] = Field(default_factory=list)
    error: str | None = None


def merge_sources(sources: list[Source]) -> list[Source]:
    return list({source.id: source for source in sources}.values())


def source_list(sources: list[Source]) -> str:
    if not sources:
        return ""
    rows = []
    for source in merge_sources(sources):
        page = f", page {source.page}" if source.page else ""
        rows.append(
            f"- [{source.id}] {source.title}{page} — {source.locator} "
            f"(retrieved {source.retrieved_at})"
        )
    return "\n\nSources retrieved:\n" + "\n".join(rows)
