"""Dev 2 (T2): busca vetorial e híbrida (BM25 + vetorial com RRF)."""
from src.schema import Hit


def search(query: str, k: int = 5, mode: str = "vector") -> list[Hit]:
    """Retorna os top-k chunks.

    mode="vector": FAISS com config.QUERY_PREFIX; Hit.score = similaridade de cosseno.
    mode="hybrid": BM25 + vetorial fundidos por RRF (config.RRF_K).
    Em ambos os modos, a recusa por limiar usa o score vetorial do melhor chunk.
    """
    raise NotImplementedError
