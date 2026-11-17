"""Testes de retrieval sem rede: encoder falso e índices FAISS pequenos."""
import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from src import retrieval
from src.retrieval import Retriever, rrf_fuse, tokenize

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "index"


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v)


def _make_index(tmp_path, chunks, vectors):
    with open(tmp_path / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    vecs = np.stack([_unit(v) for v in vectors])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / "faiss.index"))
    return tmp_path


def _chunk(i, text, doc="d.pdf"):
    return {"id": f"{doc}:{i + 1}:0", "doc": doc, "page": i + 1, "text": text}


def test_tokenize_lower_no_accents_alnum():
    assert tokenize("Pressão Arterial: 140/90 mmHg, gestação!") == [
        "pressao", "arterial", "140", "90", "mmhg", "gestacao"]


def test_rrf_order_small_case():
    # 0: ranks 1 e 3 -> 1/61 + 1/63 ; 1: ranks 2 e 1 -> 1/62 + 1/61 (maior)
    # 2: ranks 3 e 2 -> 1/63 + 1/62 ; 3: só rank 4 na primeira lista
    order = rrf_fuse([[0, 1, 2, 3], [1, 2, 0]], rrf_k=60)
    assert order == [1, 0, 2, 3]


def test_rrf_item_in_both_lists_wins():
    assert rrf_fuse([[5, 7], [9, 7]])[0] == 7


def test_vector_mode_scores_are_cosine(tmp_path):
    chunks = [_chunk(i, t) for i, t in enumerate(["alfa", "beta", "gama"])]
    d = _make_index(tmp_path, chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    q = _unit([0.8, 0.6, 0])
    r = Retriever(d, encoder=lambda _: q)
    hits, best = r.search_with_best("x", k=2, mode="vector")
    assert [h.page for h in hits] == [1, 2]
    assert hits[0].score == pytest.approx(0.8, abs=1e-6)
    assert hits[1].score == pytest.approx(0.6, abs=1e-6)
    assert best == pytest.approx(0.8, abs=1e-6)


def test_hybrid_bm25_only_chunk_gets_cosine_score(tmp_path):
    # o chunk 4 só casa por BM25 (vetor ortogonal à consulta), mas sobe ao topo via RRF
    chunks = [
        _chunk(0, "texto sobre vacinas"),
        _chunk(1, "texto sobre diabetes"),
        _chunk(2, "texto sobre gestantes"),
        _chunk(3, "monofilamento monofilamento"),
    ]
    vecs = [[1, 0.1, 0], [1, 0.2, 0], [1, 0.3, 0], [0, 0, 1]]
    d = _make_index(tmp_path, chunks, vecs)
    q = _unit([1, 0, 0])
    r = Retriever(d, encoder=lambda _: q)
    hits, best = r.search_with_best("monofilamento", k=2, mode="hybrid")
    assert hits[0].page == 4  # BM25 rank 1 + vetor rank 4 > vetor rank 1 sozinho
    assert hits[0].score == pytest.approx(0.0, abs=1e-6)  # cosseno, não score RRF
    assert best == pytest.approx(float(np.dot(_unit(vecs[0]), q)), abs=1e-6)


def test_invalid_mode(tmp_path):
    d = _make_index(tmp_path, [_chunk(0, "a")], [[1, 0]])
    with pytest.raises(ValueError):
        Retriever(d, encoder=lambda _: _unit([1, 0])).search_with_best("a", mode="xyz")


def test_fixture_index_matches_contract():
    r = Retriever(FIXTURE_DIR, encoder=lambda _: None)
    assert r.index.ntotal == len(r.chunks) > 0
    assert np.linalg.norm(r.index.reconstruct(0)) == pytest.approx(1.0, abs=1e-4)
    # consulta igual ao vetor do chunk 4 -> ele é o primeiro nos dois modos
    r._encode = lambda _: r.index.reconstruct(4)
    for mode in ("vector", "hybrid"):
        hits, best = r.search_with_best("pré-natal consultas", k=5, mode=mode)
        assert hits[0].id == r.chunks[4]["id"]
        assert best == pytest.approx(1.0, abs=1e-4)


def test_get_retriever_is_lazy_and_cached(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, d):
            calls.append(d)

    monkeypatch.setattr(retrieval, "Retriever", Fake)
    monkeypatch.setattr(retrieval, "_retrievers", {})
    assert calls == []
    a = retrieval.get_retriever(FIXTURE_DIR)
    b = retrieval.get_retriever(str(FIXTURE_DIR))
    assert a is b and len(calls) == 1
