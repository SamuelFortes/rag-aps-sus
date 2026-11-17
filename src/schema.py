"""Tipos compartilhados (contrato entre módulos)."""
from dataclasses import dataclass, field


@dataclass
class Hit:
    id: str      # "<doc>:<page>:<n>"
    doc: str     # nome do arquivo PDF
    page: int    # base 1
    text: str
    score: float


@dataclass
class Answer:
    text: str
    refused: bool
    citations: list[tuple[str, int]] = field(default_factory=list)  # (doc, page)
    sources: list[Hit] = field(default_factory=list)
    error: str | None = None
