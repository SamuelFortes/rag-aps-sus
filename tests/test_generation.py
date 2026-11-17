"""Testes de geração sem rede: busca e cliente Gemini mockados."""
import pytest

from src import config, generation
from src.generation import answer, build_prompt, parse_citations, valid_citations
from src.schema import Hit

HITS = [
    Hit(id="cab37.pdf:12:0", doc="cab37.pdf", page=12, text="PA >= 140/90.", score=0.9),
    Hit(id="cab36.pdf:5:1", doc="cab36.pdf", page=5, text="Exame dos pés anual.", score=0.85),
]


class _Resp:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.exc:
            raise self.exc
        return _Resp(self.text)


class FakeClient:
    def __init__(self, text=None, exc=None):
        self.models = FakeModels(text, exc)


@pytest.fixture
def search_ok(monkeypatch):
    monkeypatch.setattr(generation.retrieval, "search_with_best",
                        lambda q, k, mode, index_dir: (HITS, 0.95))


def test_parse_citations_variants():
    text = ("Pressão alta [cab37, p. 12]. Pés [cab36.pdf, p.5; cab37, p. 12]. "
            "Outro [cab37, pág. 3] e [nota sem página].")
    assert parse_citations(text) == [("cab37", 12), ("cab36", 5), ("cab37", 3)]


def test_valid_citations_discards_unsent_and_restores_pdf():
    text = "A [cab37, p. 12]. B [cab37, p. 99]. C [inventado, p. 1]. D [cab36, p. 5]."
    assert valid_citations(text, HITS) == [("cab37.pdf", 12), ("cab36.pdf", 5)]


def test_prompt_labels_chunks():
    p = build_prompt("Qual o limite?", HITS)
    assert "[cab37, p. 12]\nPA >= 140/90." in p
    assert "[cab36, p. 5]" in p
    assert "RECUSA" in p and "Qual o limite?" in p


def test_refuses_below_threshold_without_llm(monkeypatch):
    monkeypatch.setattr(generation.retrieval, "search_with_best",
                        lambda q, k, mode, index_dir: (HITS, config.SIMILARITY_THRESHOLD - 0.01))
    client = FakeClient("não deveria ser chamado")
    a = answer("pergunta fora do escopo", client=client)
    assert a.refused and a.text == config.RECUSA and a.error is None
    assert client.models.calls == []


def test_answer_with_citations(search_ok):
    client = FakeClient("  É 140/90 mmHg [cab37, p. 12] [falso, p. 2].  ")
    a = answer("Qual o limite?", "vector", client=client)
    assert not a.refused and a.error is None
    assert a.text == "É 140/90 mmHg [cab37, p. 12] [falso, p. 2]."
    assert a.citations == [("cab37.pdf", 12)]
    assert a.sources == HITS
    call = client.models.calls[0]
    assert call["model"] == config.GEMINI_MODEL
    assert call["config"].temperature == config.TEMPERATURE


def test_refusal_from_llm(search_ok):
    a = answer("x", client=FakeClient("  RECUSA \n"))
    assert a.refused and a.text == config.RECUSA and a.citations == [] and a.error is None


def test_api_error_becomes_answer_error(search_ok):
    a = answer("x", client=FakeClient(exc=RuntimeError("503 indisponível")))
    assert a.error and "503" in a.error and not a.refused


def test_missing_key_becomes_answer_error(search_ok, monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    a = answer("x")
    assert a.error and "GEMINI_API_KEY" in a.error


def test_retrieval_error_becomes_answer_error(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("data/index/faiss.index")

    monkeypatch.setattr(generation.retrieval, "search_with_best", boom)
    a = answer("x", client=FakeClient("ok"))
    assert a.error and "faiss.index" in a.error
