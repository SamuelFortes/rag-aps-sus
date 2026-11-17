"""Dev 2 (T2): busca vetorial e híbrida (BM25 + vetorial com RRF).

O índice (chunks.jsonl + faiss.index) e o modelo de embeddings são carregados
de forma preguiçosa na primeira busca e mantidos em cache por diretório.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

from src import config
from src.schema import Hit

# Hugging Face Spaces com ZeroGPU: o pacote `spaces` existe e o Space exige ao menos
# uma função com @spaces.GPU. Fora dos Spaces (local, testes) nada disso é ativado.
try:
    import spaces
except ImportError:
    spaces = None
ZERO_GPU = spaces is not None and os.getenv("SPACES_ZERO_GPU", "").lower() in ("1", "true")


def _gpu(fn):
    """Aplica @spaces.GPU quando o pacote `spaces` está disponível; senão, não faz nada."""
    return spaces.GPU(duration=20)(fn) if spaces is not None else fn


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Minúsculas, sem acentos, só sequências alfanuméricas."""
    norm = unicodedata.normalize("NFKD", text.lower())
    norm = "".join(c for c in norm if not unicodedata.combining(c))
    return _TOKEN_RE.findall(norm)


def rrf_fuse(rankings: list[list[int]], rrf_k: int = config.RRF_K) -> list[int]:
    """Reciprocal Rank Fusion: soma 1/(rrf_k + posição), posição base 1.

    Empates são resolvidos pela ordem de primeira aparição (estável).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda i: -scores[i])


@lru_cache(maxsize=1)
def _default_encoder():
    from sentence_transformers import SentenceTransformer

    # No ZeroGPU o modelo é colocado em "cuda" na carga; a GPU real só é alocada
    # durante as chamadas a funções com @spaces.GPU.
    return SentenceTransformer(config.EMBEDDING_MODEL, device="cuda" if ZERO_GPU else None)


def warmup(index_dir: Path | str | None = None) -> None:
    """Carrega índice e modelo de embeddings sem fazer nenhuma busca (não usa GPU)."""
    get_retriever(index_dir)
    _default_encoder()


@_gpu
def encode_query(query: str) -> np.ndarray:
    vec = _default_encoder().encode(
        [config.QUERY_PREFIX + query], normalize_embeddings=True
    )
    return np.asarray(vec, dtype="float32")


class Retriever:
    """Índice carregado de `index_dir` (chunks.jsonl + faiss.index)."""

    def __init__(self, index_dir: Path, encoder=None):
        import faiss
        from rank_bm25 import BM25Okapi

        index_dir = Path(index_dir)
        with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"faiss.index tem {self.index.ntotal} vetores, mas chunks.jsonl tem "
                f"{len(self.chunks)} linhas"
            )
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._encode = encoder or encode_query

    def _hit(self, i: int, score: float) -> Hit:
        c = self.chunks[i]
        return Hit(id=c["id"], doc=c["doc"], page=int(c["page"]), text=c["text"], score=float(score))

    def _qvec(self, query: str) -> np.ndarray:
        return np.asarray(self._encode(query), dtype="float32").reshape(1, -1)

    def _vector(self, qvec: np.ndarray, n: int) -> tuple[list[int], list[float]]:
        n = min(n, self.index.ntotal)
        scores, ids = self.index.search(qvec, n)
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def _cosine(self, qvec: np.ndarray, i: int) -> float:
        return float(np.dot(self.index.reconstruct(int(i)), qvec[0]))

    def search_with_best(
        self, query: str, k: int = config.TOP_K, mode: str = "vector"
    ) -> tuple[list[Hit], float]:
        """Retorna (hits, melhor similaridade vetorial da consulta)."""
        if mode not in config.SEARCH_MODES:
            raise ValueError(f"mode inválido: {mode!r}; use um de {config.SEARCH_MODES}")
        if not self.chunks:
            return [], float("-inf")
        qvec = self._qvec(query)
        if mode == "vector":
            ids, scores = self._vector(qvec, k)
            hits = [self._hit(i, s) for i, s in zip(ids, scores)]
            return hits, (scores[0] if scores else float("-inf"))

        # hybrid: candidatos = top-n de cada lista; n maior que k dá margem ao RRF
        n = max(k * 4, 20)
        vec_ids, vec_scores = self._vector(qvec, n)
        cos = dict(zip(vec_ids, vec_scores))
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_ids = [int(i) for i in np.argsort(-bm25_scores, kind="stable")[:n] if bm25_scores[i] > 0]
        fused = rrf_fuse([vec_ids, bm25_ids])[:k]
        hits = [self._hit(i, cos[i] if i in cos else self._cosine(qvec, i)) for i in fused]
        return hits, (vec_scores[0] if vec_scores else float("-inf"))


_retrievers: dict[Path, Retriever] = {}


def get_retriever(index_dir: Path | str | None = None) -> Retriever:
    """Retriever em cache por diretório (padrão: config.INDEX_DIR)."""
    key = Path(index_dir or config.INDEX_DIR).resolve()
    if key not in _retrievers:
        _retrievers[key] = Retriever(key)
    return _retrievers[key]


def search_with_best(
    query: str, k: int = config.TOP_K, mode: str = "vector", index_dir: Path | str | None = None
) -> tuple[list[Hit], float]:
    return get_retriever(index_dir).search_with_best(query, k, mode)


def search(
    query: str, k: int = config.TOP_K, mode: str = "vector", index_dir: Path | str | None = None
) -> list[Hit]:
    """Retorna os top-k chunks.

    mode="vector": FAISS com config.QUERY_PREFIX; Hit.score = similaridade de cosseno.
    mode="hybrid": BM25 + vetorial fundidos por RRF (config.RRF_K); Hit.score continua
    sendo o cosseno vetorial do chunk (mesma escala do limiar de recusa).
    """
    return search_with_best(query, k, mode, index_dir)[0]


def best_similarity(query: str, index_dir: Path | str | None = None) -> float:
    """Melhor similaridade de cosseno da consulta contra todo o índice."""
    return search_with_best(query, 1, "vector", index_dir)[1]
