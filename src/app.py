"""Dev 2 (T5): interface Gradio.

Uma pergunta por vez, sem histórico. Usa apenas data/index/ (não acessa PDFs).
Executar localmente: python app.py (na raiz) ou python -m src.app.
"""
from __future__ import annotations

import logging
import sys

import gradio as gr

from src import config, generation, retrieval
from src.schema import Answer

log = logging.getLogger("assistente_aps")

AVISO = ("Projeto de demonstração. Não substitui orientação profissional nem os "
         "documentos oficiais do Ministério da Saúde.")
ERRO_AMIGAVEL = "Não foi possível gerar a resposta agora. Tente novamente em instantes."
TRECHO_MAX = 300

DOCUMENTOS = [
    ("carteira_servicos_atencao_primaria_saude.pdf", "Carteira de Serviços da Atenção Primária à Saúde"),
    ("politica_nacional_atencao_basica.pdf", "Política Nacional de Atenção Básica (PNAB)"),
    ("cadernos_atencao_basica_32_prenatal.pdf", "Cadernos de Atenção Básica nº 32 — Atenção ao Pré-Natal de Baixo Risco"),
    ("estrategias_cuidado_pessoa_diabetes_mellitus_cab36.pdf",
     "Cadernos de Atenção Básica nº 36 — Estratégias para o Cuidado da Pessoa com Doença Crônica: Diabetes Mellitus"),
    ("hipertensao_arterial_sistemica_cab37.pdf",
     "Cadernos de Atenção Básica nº 37 — Estratégias para o Cuidado da Pessoa com Doença Crônica: Hipertensão Arterial Sistêmica"),
]

EXEMPLOS = [
    "Quantas consultas de pré-natal são recomendadas para uma gestante de baixo risco?",
    "Com que frequência deve ser feito o exame dos pés da pessoa com diabetes?",
    "Quais valores de pressão arterial definem hipertensão no consultório?",
    "Como configuro uma VPN no Windows 11?",
]


def _trecho(text: str, limit: int = TRECHO_MAX) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def format_sources(ans: Answer) -> str:
    """Markdown com as fontes usadas; vazio em recusa ou erro."""
    if ans.refused or ans.error or not ans.sources:
        return ""
    linhas = ["### Fontes consultadas"]
    for h in ans.sources:
        linhas.append(f"- **{generation.doc_stem(h.doc)}, p. {h.page}** — {_trecho(h.text)}")
    return "\n".join(linhas)


def format_answer(ans: Answer) -> tuple[str, str]:
    """(resposta em Markdown, fontes em Markdown)."""
    if ans.error:
        log.error("Falha ao gerar resposta: %s", ans.error)
        return f"⚠️ {ERRO_AMIGAVEL}", ""
    return ans.text, format_sources(ans)


def responder(pergunta: str) -> tuple[str, str]:
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return "Digite uma pergunta.", ""
    try:
        ans = generation.answer(pergunta, config.DEFAULT_MODE)
    except Exception:  # answer não deve lançar, mas a interface nunca mostra traceback
        log.exception("Erro inesperado")
        return f"⚠️ {ERRO_AMIGAVEL}", ""
    return format_answer(ans)


def preload() -> None:
    """Carrega índice e modelo de embeddings antes da primeira pergunta."""
    try:
        retrieval.search_with_best("pré-carregamento", 1, config.DEFAULT_MODE)
        log.info("Índice e modelo de embeddings carregados.")
    except Exception:
        log.exception("Falha ao pré-carregar o índice")


def build_demo() -> gr.Blocks:
    rodape = "**Documentos consultados (Ministério da Saúde):**\n\n" + "\n".join(
        f"- {titulo}" for _, titulo in DOCUMENTOS)
    with gr.Blocks(title="Assistente da Atenção Primária (SUS)") as demo:
        gr.Markdown(
            "# Assistente da Atenção Primária (SUS)\n"
            "Pergunte sobre pré-natal, diabetes, hipertensão, PNAB e a Carteira de Serviços da APS. "
            "As respostas citam o documento e a página de origem.")
        gr.Markdown(f"> ⚠️ **{AVISO}**")
        pergunta = gr.Textbox(label="Pergunta", lines=2,
                              placeholder="Ex.: Quantas consultas de pré-natal são recomendadas?")
        botao = gr.Button("Perguntar", variant="primary")
        resposta = gr.Markdown(label="Resposta")
        fontes = gr.Markdown()
        gr.Examples(examples=[[e] for e in EXEMPLOS], inputs=pergunta,
                    label="Perguntas de exemplo")
        gr.Markdown(rodape)
        botao.click(responder, inputs=pergunta, outputs=[resposta, fontes])
        pergunta.submit(responder, inputs=pergunta, outputs=[resposta, fontes])
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    preload()
    build_demo().queue().launch()


if __name__ == "__main__":
    main()
