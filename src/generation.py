"""Dev 2 (T2): geração com Gemini, citação inline e recusa em duas camadas."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from src import config, retrieval
from src.schema import Answer, Hit

PROMPT_TEMPLATE = """Você é um assistente para profissionais da Atenção Primária à Saúde (SUS).
Responda à pergunta usando SOMENTE as informações do CONTEXTO abaixo.

Regras:
- Responda em português do Brasil, de forma objetiva.
- Cite a fonte de cada afirmação logo após ela, exatamente no formato [documento, p. N], copiando o rótulo do trecho usado (por exemplo: {exemplo}).
- Não use conhecimento externo ao contexto e não invente citações.
- Se o contexto não contiver a resposta, responda EXATAMENTE com a palavra RECUSA e nada mais.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA:"""

# "[doc, p. 12]" ou "[doc, p. 12; outro, p. 3]"
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_CITE_RE = re.compile(r"^\s*(.+?)\s*,\s*p(?:ág|ag)?\.?\s*(\d+)\s*$", re.IGNORECASE)


def doc_stem(doc: str) -> str:
    return doc[:-4] if doc.lower().endswith(".pdf") else doc


def label(hit: Hit) -> str:
    return f"[{doc_stem(hit.doc)}, p. {hit.page}]"


def build_prompt(question: str, hits: list[Hit]) -> str:
    contexto = "\n\n".join(f"{label(h)}\n{h.text.strip()}" for h in hits)
    exemplo = label(hits[0]) if hits else "[documento, p. 1]"
    return PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question.strip(), exemplo=exemplo)


def parse_citations(text: str) -> list[tuple[str, int]]:
    """Extrai (doc sem .pdf, página) das citações inline, em ordem e sem repetição."""
    out: list[tuple[str, int]] = []
    for bracket in _BRACKET_RE.findall(text):
        for part in bracket.split(";"):
            m = _CITE_RE.match(part)
            if m:
                cite = (doc_stem(m.group(1).strip()), int(m.group(2)))
                if cite not in out:
                    out.append(cite)
    return out


def valid_citations(text: str, hits: list[Hit]) -> list[tuple[str, int]]:
    """Citações do texto que correspondem a chunks enviados, como (doc com .pdf, página)."""
    sent = {(doc_stem(h.doc), h.page): h.doc for h in hits}
    return [(sent[c], c[1]) for c in parse_citations(text) if c in sent]


REQUEST_TIMEOUT_MS = 60_000  # sem timeout, uma chamada travada bloquearia a interface


def _make_client():
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def _generate(client, prompt: str) -> str:
    from google.genai import types

    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=config.TEMPERATURE),
    )
    return resp.text or ""


def answer(
    question: str,
    mode: str = config.DEFAULT_MODE,
    *,
    index_dir: Path | str | None = None,
    client=None,
) -> Answer:
    """Responde em português com citações "[<doc sem .pdf>, p. N]".

    Recusa (Answer.refused=True, Answer.text == config.RECUSA) quando a melhor
    similaridade vetorial < config.SIMILARITY_THRESHOLD (sem chamar o LLM) ou
    quando o LLM indica que o contexto não responde. Erros de API/chave ausente
    preenchem Answer.error, sem lançar exceção.
    """
    try:
        hits, best = retrieval.search_with_best(question, config.TOP_K, mode, index_dir)
    except Exception as e:  # índice ausente, modelo indisponível etc.
        return Answer(text="", refused=False, error=f"Erro na busca: {e}")

    if not hits or best < config.SIMILARITY_THRESHOLD:
        return Answer(text=config.RECUSA, refused=True, sources=hits)

    try:
        if client is None:
            if not config.GEMINI_API_KEY:
                return Answer(text="", refused=False, sources=hits,
                              error="GEMINI_API_KEY não configurada.")
            client = _make_client()
        text = _generate(client, build_prompt(question, hits)).strip()
    except Exception as e:
        return Answer(text="", refused=False, sources=hits,
                      error=f"Erro na API do Gemini: {type(e).__name__}: {e}")

    if text == "RECUSA" or text == config.RECUSA:
        return Answer(text=config.RECUSA, refused=True, sources=hits)
    return Answer(text=text, refused=False, citations=valid_citations(text, hits), sources=hits)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pergunta ao assistente RAG.")
    p.add_argument("question")
    p.add_argument("--mode", default=config.DEFAULT_MODE, choices=config.SEARCH_MODES)
    p.add_argument("--index-dir", default=None)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    a = answer(args.question, args.mode, index_dir=args.index_dir)
    print(f"refused={a.refused} error={a.error}")
    print(a.text)
    print("citations:", a.citations)
    for h in a.sources:
        print(f"  {h.score:.3f} {h.id}")
