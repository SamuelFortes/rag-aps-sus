"""Testes da formatação da interface, com generation.answer mockado (sem rede)."""
import logging

import pytest

from src import app, config
from src.schema import Answer, Hit

HITS = [
    Hit(id="cab37.pdf:34:0", doc="hipertensao_arterial_sistemica_cab37.pdf", page=34,
        text="Valores   de PA no consultório\n>= 140/90 mmHg caracterizam hipertensão.", score=0.88),
    Hit(id="x.pdf:2:0", doc="politica_nacional_atencao_basica.pdf", page=2,
        text="palavra " * 100, score=0.85),
]


@pytest.fixture
def mock_answer(monkeypatch):
    def set_(ans):
        calls = []

        def fake(question, mode):
            calls.append((question, mode))
            return ans

        monkeypatch.setattr(app.generation, "answer", fake)
        return calls

    return set_


def test_cited_answer_shows_text_and_sources(mock_answer):
    texto = "PA >= 140/90 mmHg [hipertensao_arterial_sistemica_cab37, p. 34]."
    calls = mock_answer(Answer(text=texto, refused=False,
                               citations=[("hipertensao_arterial_sistemica_cab37.pdf", 34)],
                               sources=HITS))
    resposta, fontes = app.responder("  Qual o valor?  ")
    assert calls == [("Qual o valor?", config.DEFAULT_MODE)]
    assert resposta == texto
    assert "**hipertensao_arterial_sistemica_cab37, p. 34** — Valores de PA no consultório >= 140/90" in fontes
    assert ".pdf" not in fontes
    assert "**politica_nacional_atencao_basica, p. 2**" in fontes


def test_long_excerpt_truncated():
    trecho = app._trecho("palavra " * 100)
    assert len(trecho) <= app.TRECHO_MAX + 1 and trecho.endswith("…")


def test_refusal_hides_sources(mock_answer):
    mock_answer(Answer(text=config.RECUSA, refused=True, sources=HITS))
    resposta, fontes = app.responder("Como configuro uma VPN?")
    assert resposta == config.RECUSA
    assert fontes == ""


def test_error_is_friendly_and_logged(mock_answer, caplog):
    mock_answer(Answer(text="", refused=False, sources=HITS,
                       error="Erro na API do Gemini: ServerError: 503 UNAVAILABLE"))
    with caplog.at_level(logging.ERROR, logger="assistente_aps"):
        resposta, fontes = app.responder("x")
    assert app.ERRO_AMIGAVEL in resposta and "503" not in resposta
    assert fontes == ""
    assert "503 UNAVAILABLE" in caplog.text


def test_unexpected_exception_never_reaches_ui(monkeypatch):
    def boom(*a):
        raise RuntimeError("traceback interno")

    monkeypatch.setattr(app.generation, "answer", boom)
    resposta, fontes = app.responder("x")
    assert app.ERRO_AMIGAVEL in resposta and "traceback" not in resposta


def test_empty_question_does_not_call_answer(mock_answer):
    calls = mock_answer(Answer(text="nunca", refused=False))
    resposta, _ = app.responder("   ")
    assert calls == [] and "Digite" in resposta


def test_build_demo_constructs():
    assert app.build_demo() is not None
