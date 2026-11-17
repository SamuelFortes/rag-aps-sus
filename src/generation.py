"""Dev 2 (T2): geração com Gemini, citação inline e recusa em duas camadas."""
from src.schema import Answer


def answer(question: str, mode: str = "hybrid") -> Answer:
    """Responde em português com citações "[<doc sem .pdf>, p. N]".

    Recusa (Answer.refused=True, Answer.text == config.RECUSA) quando a melhor
    similaridade vetorial < config.SIMILARITY_THRESHOLD (sem chamar o LLM) ou
    quando o LLM indica que o contexto não responde. Erros de API/chave ausente
    preenchem Answer.error, sem lançar exceção.
    """
    raise NotImplementedError
