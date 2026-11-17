"""Testes do cálculo de métricas da avaliação (T3a), com search/answer mockados."""
import json

import pytest

from eval import run_eval as ev
from src import config
from src.schema import Answer, Hit

QUESTIONS = [
    {"id": "q1", "question": "Q1", "in_scope": True, "doc": "a.pdf", "pages": [3, 4]},
    {"id": "q2", "question": "Q2", "in_scope": True, "doc": "b.pdf", "pages": [10]},
    {"id": "q3", "question": "Q3", "in_scope": True, "doc": "c.pdf", "pages": [1]},
    {"id": "q4", "question": "Q4", "in_scope": True, "doc": "d.pdf", "pages": [7]},
    {"id": "f1", "question": "F1", "in_scope": False, "doc": None, "pages": []},
    {"id": "f2", "question": "F2", "in_scope": False, "doc": None, "pages": []},
]


def h(doc, page, score):
    return Hit(f"{doc}:{page}:0", doc, page, "txt", score)


SEARCH = {
    # q1: acerto na 5ª posição (página alternativa 4)
    "Q1": [h("x.pdf", 1, .9), h("x.pdf", 2, .89), h("x.pdf", 3, .88), h("a.pdf", 9, .87),
           h("a.pdf", 4, .86)],
    # q2: acerto só na 6ª posição -> não conta no hit@5
    "Q2": [h("x.pdf", 1, .9)] * 5 + [h("b.pdf", 10, .8)],
    # q3: doc certo, página errada
    "Q3": [h("c.pdf", 2, .84)],
    # q4: acerto na 1ª
    "Q4": [h("d.pdf", 7, .83)],
    "F1": [h("x.pdf", 1, .75)],
    "F2": [h("x.pdf", 1, .81)],
}

ANSWERS = {
    "Q1": Answer("r [a, p. 4]", False, citations=[("a", 4)]),            # sem .pdf: correta
    "Q2": Answer("r [b, p. 11]", False, citations=[("b.pdf", 11)]),       # página errada
    "Q3": Answer(config.RECUSA, True),                                    # recusa indevida
    "Q4": Answer("", False, error="429 RESOURCE_EXHAUSTED"),              # erro de API
    "F1": Answer(config.RECUSA, True),                                    # recusa correta
    "F2": Answer("inventado [x, p. 1]", False, citations=[("x", 1)]),     # não recusou
}


def search_mock(query, k=5, mode="vector"):
    return SEARCH[query][:k]


def answer_mock(question, mode="hybrid"):
    return ANSWERS[question]


def no_sleep(_):
    pass


def test_metrics_full_run():
    rows, m = ev.run_eval(QUESTIONS, "vector", search_mock, answer_mock,
                          retries=0, sleep_fn=no_sleep)
    assert m["n_in_scope"] == 4 and m["n_out_of_scope"] == 2
    assert m["hit@5"] == pytest.approx(2 / 4)  # q1 e q4
    by_id = {r["id"]: r for r in rows}
    assert by_id["q1"]["hit@5"] is True
    assert by_id["q2"]["hit@5"] is False
    assert by_id["q3"]["hit@5"] is False
    assert by_id["q4"]["hit@5"] is True
    assert m["citation_correct"] == pytest.approx(1 / 4)
    assert m["refusal_correct"] == pytest.approx(1 / 2)
    assert m["wrong_refusals"] == 1
    assert m["api_errors"] == 1
    assert by_id["f1"]["hit@5"] is None and by_id["f1"]["refusal_correct"] is True
    assert by_id["q1"]["hits"][4] == {"doc": "a.pdf", "page": 4, "score": 0.86, "id": "a.pdf:4:0"}


def test_hit_at_5_counts_only_top_5():
    # SEARCH["Q2"] tem o acerto na posição 6; mesmo pedindo k maior, hit@5 ignora
    assert ev.hit_at_k(SEARCH["Q2"], QUESTIONS[1]) is False
    assert ev.hit_at_k(SEARCH["Q2"], QUESTIONS[1], k=6) is True


def test_retrieval_only_does_not_call_answer_and_sweeps_thresholds():
    def boom(*a, **kw):
        raise AssertionError("answer não deveria ser chamado")

    rows, m = ev.run_eval(QUESTIONS, "vector", search_mock, boom, retrieval_only=True,
                          thresholds=[0.80, 0.85])
    assert "citation_correct" not in m
    assert {r["id"]: r["best_vector_score"] for r in rows}["q3"] == pytest.approx(0.84)
    s80, s85 = m["threshold_sweep"]
    # melhores: q1 .9, q2 .9, q3 .84, q4 .83 | f1 .75, f2 .81
    assert (s80["in_scope_refused"], s80["out_of_scope_refused"]) == (0, 1)
    assert (s85["in_scope_refused"], s85["out_of_scope_refused"]) == (2, 2)


def test_hybrid_best_similarity_uses_vector_search():
    calls = []

    def search(query, k=5, mode="vector"):
        calls.append(mode)
        if mode == "hybrid":
            return [h("a.pdf", 3, 0.032)]
        return [h("a.pdf", 3, 0.91)]

    rows, m = ev.run_eval(QUESTIONS[:1], "hybrid", search, retrieval_only=True)
    assert calls == ["hybrid", "vector"]
    assert rows[0]["best_vector_score"] == pytest.approx(0.91)
    assert rows[0]["hits"][0]["score"] == pytest.approx(0.032)
    assert m["hit@5"] == 1.0


def test_retry_on_429_then_success():
    responses = [Answer("", False, error="429 Too Many Requests"),
                 Answer("", False, error="RESOURCE_EXHAUSTED"),
                 Answer("ok [a, p. 3]", False, citations=[("a", 3)])]
    waits = []

    def answer(q, mode="hybrid"):
        return responses.pop(0)

    ans = ev.call_answer_with_retry(answer, "Q1", "vector", retries=3, retry_wait=10,
                                    sleep_fn=waits.append)
    assert ans.error is None and ans.citations == [("a", 3)]
    assert waits == [10, 20]


def test_retry_gives_up_and_exceptions_become_errors():
    def answer(q, mode="hybrid"):
        raise RuntimeError("429 quota exceeded")

    waits = []
    ans = ev.call_answer_with_retry(answer, "Q", "vector", retries=2, retry_wait=1,
                                    sleep_fn=waits.append)
    assert "429" in ans.error and waits == [1, 2]


def test_non_429_error_is_not_retried():
    calls = []

    def answer(q, mode="hybrid"):
        calls.append(1)
        return Answer("", False, error="GEMINI_API_KEY ausente")

    ans = ev.call_answer_with_retry(answer, "Q", "vector", retries=3, sleep_fn=no_sleep)
    assert len(calls) == 1 and ans.error


def test_sleep_between_calls():
    waits = []
    ev.run_eval(QUESTIONS[:3], "vector", search_mock,
                lambda q, mode: Answer("", False), sleep=4.0, sleep_fn=waits.append)
    assert waits == [4.0, 4.0]


def test_outputs_accumulate_modes_and_table(tmp_path):
    for mode in ("vector", "hybrid"):
        rows, m = ev.run_eval(QUESTIONS, mode, search_mock, answer_mock, retries=0,
                              sleep_fn=no_sleep)
        ev.save_outputs(tmp_path, mode, rows, m, retrieval_only=False)
    rows, m = ev.run_eval(QUESTIONS, "vector", search_mock, retrieval_only=True)
    ev.save_outputs(tmp_path, "vector", rows, m, retrieval_only=True)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"vector", "hybrid", "retrieval_only"}
    assert summary["vector"]["citation_correct"] == pytest.approx(0.25)
    assert (tmp_path / "vector.jsonl").exists() and (tmp_path / "hybrid.jsonl").exists()
    assert (tmp_path / "vector_retrieval.jsonl").exists()
    line = json.loads((tmp_path / "vector.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for key in ("id", "hits", "answer", "citations", "refused", "hit@5", "citation_correct"):
        assert key in line

    table = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert "| Baseline (vetorial) | 50.0% | 25.0% | 50.0% | 1/4 | 1 |" in table
    assert "Híbrida" in table


def test_load_example_questions():
    qs = ev.load_questions(config.EVAL_DIR / "questions_example.jsonl")
    assert len(qs) == 4
    assert sum(q["in_scope"] for q in qs) == 3


def test_mock_cli_produces_table(tmp_path, capsys):
    args = ["--questions", str(config.EVAL_DIR / "questions_example.jsonl"),
            "--results-dir", str(tmp_path), "--mock"]
    assert ev.main(["--mode", "vector", *args]) == 0
    assert ev.main(["--mode", "hybrid", *args]) == 0
    table = (tmp_path / "table.md").read_text(encoding="utf-8")
    assert "| Baseline (vetorial) | 66.7% | 66.7% | 100.0% | 0/3 | 0 |" in table
    assert "| Híbrida (BM25 + vetorial, RRF) | 100.0% | 100.0% | 100.0% | 0/3 | 0 |" in table
