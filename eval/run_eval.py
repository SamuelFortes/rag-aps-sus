"""Dev 3 (T3a): avaliação hit@5, citação correta e recusa correta.

Uso:
    python -m eval.run_eval --mode vector|hybrid [--retrieval-only] [--questions <path>]
        [--thresholds 0.70,0.75,...] [--sleep 4] [--retries 3] [--retry-wait 30]
        [--results-dir <dir>] [--mock]

Saídas (em --results-dir, padrão config.RESULTS_DIR):
    <modo>.jsonl            resultado por pergunta (execução completa, com Gemini)
    <modo>_retrieval.jsonl  resultado por pergunta (--retrieval-only)
    summary.json            métricas agregadas, acumuladas por modo
    table.md                tabela Markdown baseline (vector) x híbrida

`search_fn` e `answer_fn` são injetáveis para permitir testes com mocks; por
padrão usam src.retrieval.search e src.generation.answer. `--mock` usa um
backend falso determinístico (só para demonstrar o pipeline sem índice/API).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src import config
from src.schema import Answer, Hit

SearchFn = Callable[..., list[Hit]]
AnswerFn = Callable[..., Answer]

MODE_LABELS = {"vector": "Baseline (vetorial)", "hybrid": "Híbrida (BM25 + vetorial, RRF)"}
DEFAULT_THRESHOLDS = [round(0.70 + 0.01 * i, 2) for i in range(21)]  # 0.70..0.90


# --------------------------------------------------------------------------- io
def load_questions(path: Path) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            for key in ("id", "question", "in_scope"):
                if key not in q:
                    raise ValueError(f"{path}:{n}: campo obrigatório ausente: {key}")
            q.setdefault("doc", None)
            q.setdefault("pages", [])
            if q["in_scope"] and (not q["doc"] or not q["pages"]):
                raise ValueError(f"{path}:{n}: pergunta no escopo sem doc/pages")
            questions.append(q)
    return questions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_summary(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------- matching
def _norm_doc(doc: str | None) -> str:
    """Compara docs ignorando '.pdf' e caixa (a citação no texto vem sem .pdf)."""
    d = (doc or "").strip().lower()
    return d[:-4] if d.endswith(".pdf") else d


def matches(doc: str, page: int, q: dict) -> bool:
    return _norm_doc(doc) == _norm_doc(q.get("doc")) and int(page) in {int(p) for p in q["pages"]}


def hit_at_k(hits: list[Hit], q: dict, k: int = config.TOP_K) -> bool:
    return any(matches(h.doc, h.page, q) for h in hits[:k])


def citation_correct(ans: Answer, q: dict) -> bool:
    return any(matches(doc, page, q) for doc, page in ans.citations)


# ---------------------------------------------------------------- rate limit
def _is_rate_limit(msg: str | None) -> bool:
    """Erros transitórios que valem retry: 429 (limite de taxa) e 503 (modelo
    sobrecarregado, "UNAVAILABLE"), que o Gemini devolve em picos de demanda."""
    m = (msg or "").lower()
    return ("429" in m or "resource_exhausted" in m or "rate limit" in m or "quota" in m
            or "503" in m or "unavailable" in m)


def call_answer_with_retry(
    answer_fn: AnswerFn,
    question: str,
    mode: str,
    retries: int = 3,
    retry_wait: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Answer:
    """Chama answer_fn com retry simples em erro 429 (exceção ou Answer.error)."""
    attempt = 0
    while True:
        try:
            ans = answer_fn(question, mode=mode)
            err = ans.error
        except Exception as e:  # answer() não deveria lançar, mas por segurança
            ans, err = None, f"{type(e).__name__}: {e}"
        if not (err and _is_rate_limit(err)) or attempt >= retries:
            if ans is None:
                ans = Answer(text="", refused=False, error=err)
            return ans
        attempt += 1
        wait = retry_wait * attempt
        print(f"  429/limite de taxa; nova tentativa {attempt}/{retries} em {wait:.0f}s",
              file=sys.stderr)
        sleep_fn(wait)


# ------------------------------------------------------------------- avaliação
def _hits_to_rows(hits: list[Hit]) -> list[dict]:
    return [{"doc": h.doc, "page": h.page, "score": round(float(h.score), 6), "id": h.id}
            for h in hits]


def best_vector_similarity(search_fn: SearchFn, question: str, mode: str,
                           hits: list[Hit]) -> float | None:
    """Melhor similaridade vetorial (cosseno), a mesma que generation.answer()
    compara com o limiar. Hit.score é sempre o cosseno, mas no modo híbrido o
    top-5 fundido por RRF pode não conter o chunk de maior cosseno; por isso faz
    uma busca vetorial à parte (local, sem custo de API)."""
    vec_hits = hits if mode == "vector" else search_fn(question, k=config.TOP_K, mode="vector")
    return max((float(h.score) for h in vec_hits), default=None)


def evaluate_question(
    q: dict,
    mode: str,
    search_fn: SearchFn,
    answer_fn: AnswerFn | None,
    retrieval_only: bool,
    retries: int = 3,
    retry_wait: float = 30.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    row: dict = {"id": q["id"], "question": q["question"], "in_scope": q["in_scope"],
                 "doc": q.get("doc"), "pages": q.get("pages", [])}
    try:
        hits = search_fn(q["question"], k=config.TOP_K, mode=mode)
        row["hits"] = _hits_to_rows(hits)
        row["best_vector_score"] = best_vector_similarity(search_fn, q["question"], mode, hits)
        row["hit@5"] = hit_at_k(hits, q) if q["in_scope"] else None
        row["search_error"] = None
    except Exception as e:
        row.update(hits=[], best_vector_score=None, search_error=f"{type(e).__name__}: {e}")
        row["hit@5"] = False if q["in_scope"] else None

    if retrieval_only:
        return row

    ans = call_answer_with_retry(answer_fn, q["question"], mode, retries, retry_wait, sleep_fn)
    row["answer"] = ans.text
    row["citations"] = [[doc, page] for doc, page in ans.citations]
    row["refused"] = bool(ans.refused)
    row["error"] = ans.error
    if q["in_scope"]:
        row["citation_correct"] = citation_correct(ans, q)
        row["wrong_refusal"] = bool(ans.refused)
        row["refusal_correct"] = None
    else:
        row["citation_correct"] = None
        row["wrong_refusal"] = None
        row["refusal_correct"] = bool(ans.refused)
    return row


def _frac(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_metrics(rows: list[dict], retrieval_only: bool) -> dict:
    ins = [r for r in rows if r["in_scope"]]
    outs = [r for r in rows if not r["in_scope"]]
    m = {
        "n_questions": len(rows),
        "n_in_scope": len(ins),
        "n_out_of_scope": len(outs),
        "hit@5": _frac([bool(r["hit@5"]) for r in ins]),
        "search_errors": sum(1 for r in rows if r.get("search_error")),
    }
    if retrieval_only:
        return m
    m.update({
        "citation_correct": _frac([bool(r["citation_correct"]) for r in ins]),
        "refusal_correct": _frac([bool(r["refusal_correct"]) for r in outs]),
        "wrong_refusals": sum(1 for r in ins if r["wrong_refusal"]),
        "api_errors": sum(1 for r in rows if r.get("error")),
    })
    return m


def threshold_sweep(rows: list[dict], thresholds: list[float]) -> list[dict]:
    """Para cada limiar, quantas perguntas seriam recusadas (melhor sim. < limiar).
    Sem score (erro de busca) conta como recusada."""
    out = []
    for t in thresholds:
        def refused(r):
            s = r.get("best_vector_score")
            return s is None or s < t
        ins = [r for r in rows if r["in_scope"]]
        outs = [r for r in rows if not r["in_scope"]]
        out.append({
            "threshold": t,
            "in_scope_refused": sum(1 for r in ins if refused(r)),
            "n_in_scope": len(ins),
            "out_of_scope_refused": sum(1 for r in outs if refused(r)),
            "n_out_of_scope": len(outs),
        })
    return out


def run_eval(
    questions: list[dict],
    mode: str,
    search_fn: SearchFn,
    answer_fn: AnswerFn | None = None,
    retrieval_only: bool = False,
    sleep: float = 0.0,
    retries: int = 3,
    retry_wait: float = 30.0,
    thresholds: list[float] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Avalia todas as perguntas. Retorna (linhas por pergunta, métricas)."""
    if mode not in config.SEARCH_MODES:
        raise ValueError(f"modo inválido: {mode}")
    if not retrieval_only and answer_fn is None:
        raise ValueError("answer_fn é obrigatório sem --retrieval-only")
    rows = []
    for i, q in enumerate(questions):
        if not retrieval_only and i > 0 and sleep > 0:
            sleep_fn(sleep)  # pausa entre chamadas ao Gemini
        row = evaluate_question(q, mode, search_fn, answer_fn, retrieval_only,
                                retries, retry_wait, sleep_fn)
        rows.append(row)
        if verbose:
            flag = row.get("hit@5")
            extra = "" if retrieval_only else f" refused={row['refused']} err={bool(row['error'])}"
            print(f"[{i + 1}/{len(questions)}] {q['id']} hit@5={flag}{extra}", file=sys.stderr)
    metrics = compute_metrics(rows, retrieval_only)
    if retrieval_only:
        metrics["threshold_sweep"] = threshold_sweep(rows, thresholds or DEFAULT_THRESHOLDS)
    return rows, metrics


# ---------------------------------------------------------------- relatórios
def _pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def render_table(summary: dict) -> str:
    """Tabela baseline (vector) x híbrida. Usa a execução completa do modo; se só
    houver --retrieval-only, mostra o hit@5 dela e '—' nas demais métricas."""
    lines = [
        "| Configuração | hit@5 | Citação correta | Recusa correta | Recusas indevidas | Erros de API |",
        "|---|---|---|---|---|---|",
    ]
    for mode in config.SEARCH_MODES:
        full = summary.get(mode)
        ret = summary.get("retrieval_only", {}).get(mode)
        m = full or ret
        if not m:
            lines.append(f"| {MODE_LABELS[mode]} | — | — | — | — | — |")
            continue
        n_in, n_out = m.get("n_in_scope"), m.get("n_out_of_scope")
        wr = m.get("wrong_refusals")
        lines.append(
            f"| {MODE_LABELS[mode]} | {_pct(m.get('hit@5'))} | {_pct(m.get('citation_correct'))}"
            f" | {_pct(m.get('refusal_correct'))}"
            f" | {'—' if wr is None else f'{wr}/{n_in}'}"
            f" | {'—' if m.get('api_errors') is None else m['api_errors']} |"
        )
    n_in = n_out = None
    for mode in config.SEARCH_MODES:
        m = summary.get(mode) or summary.get("retrieval_only", {}).get(mode)
        if m:
            n_in, n_out = m["n_in_scope"], m["n_out_of_scope"]
            break
    note = ""
    if n_in is not None:
        note = (f"\n\nhit@5 e citação correta sobre {n_in} perguntas no escopo; "
                f"recusa correta sobre {n_out} perguntas fora do escopo.\n")
    return "\n".join(lines) + note


def render_sweep(sweep: list[dict]) -> str:
    lines = ["| Limiar | No escopo recusadas (indevidas) | Fora do escopo recusadas (corretas) |",
             "|---|---|---|"]
    for s in sweep:
        lines.append(f"| {s['threshold']:.2f} | {s['in_scope_refused']}/{s['n_in_scope']}"
                     f" | {s['out_of_scope_refused']}/{s['n_out_of_scope']} |")
    return "\n".join(lines)


def save_outputs(results_dir: Path, mode: str, rows: list[dict], metrics: dict,
                 retrieval_only: bool) -> dict:
    name = f"{mode}_retrieval.jsonl" if retrieval_only else f"{mode}.jsonl"
    write_jsonl(results_dir / name, rows)
    summary_path = results_dir / "summary.json"
    summary = load_summary(summary_path)
    entry = dict(metrics, updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if retrieval_only:
        summary.setdefault("retrieval_only", {})[mode] = entry
    else:
        summary[mode] = entry
    save_summary(summary_path, summary)
    (results_dir / "table.md").write_text(render_table(summary), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------- mock
def make_mock_backend(questions: list[dict]):
    """Backend falso determinístico, só para demonstrar o pipeline sem índice/API."""
    by_text = {q["question"]: q for q in questions}

    def search(query: str, k: int = 5, mode: str = "vector") -> list[Hit]:
        q = by_text.get(query, {})
        if not q.get("in_scope"):
            return [Hit(f"outro.pdf:1:{i}", "outro.pdf", 1, "...", 0.70 - 0.01 * i)
                    for i in range(k)]
        idx = questions.index(q)
        # vector acerta as perguntas de índice par; hybrid acerta todas
        ok = mode == "hybrid" or idx % 2 == 0
        page = q["pages"][0] if ok else q["pages"][0] + 50
        score = 0.031 if mode == "hybrid" else 0.86
        return [Hit(f"{q['doc']}:{page}:0", q["doc"], page, "...", score)] + [
            Hit(f"outro.pdf:9:{i}", "outro.pdf", 9, "...", score - 0.01 * (i + 1))
            for i in range(k - 1)]

    def answer(question: str, mode: str = "hybrid") -> Answer:
        q = by_text.get(question, {})
        if not q.get("in_scope"):
            return Answer(text=config.RECUSA, refused=True)
        hits = search(question, mode=mode)
        doc = hits[0].doc.removesuffix(".pdf")
        return Answer(text=f"Resposta [{doc}, p. {hits[0].page}]", refused=False,
                      citations=[(doc, hits[0].page)], sources=hits)

    return search, answer


# ------------------------------------------------------------------------ cli
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Avaliação do RAG (hit@5, citação, recusa).")
    p.add_argument("--mode", choices=config.SEARCH_MODES, required=True)
    p.add_argument("--retrieval-only", action="store_true",
                   help="só busca: hit@5, melhor similaridade vetorial e varredura de limiares")
    p.add_argument("--questions", type=Path, default=config.QUESTIONS_FILE)
    p.add_argument("--results-dir", type=Path, default=None,
                   help=f"padrão: {config.RESULTS_DIR} (com --mock: <results>/mock)")
    p.add_argument("--thresholds", type=str, default=None,
                   help="limiares separados por vírgula para a varredura (padrão 0.70..0.90)")
    p.add_argument("--sleep", type=float, default=4.0, help="pausa (s) entre chamadas ao Gemini")
    p.add_argument("--retries", type=int, default=3, help="tentativas extras em erro 429")
    p.add_argument("--retry-wait", type=float, default=30.0,
                   help="espera base (s) no retry de 429; cresce linearmente")
    p.add_argument("--mock", action="store_true",
                   help="usa backend falso (sem índice/API) para testar o pipeline")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # acentos corretos em pipes no Windows
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    questions = load_questions(args.questions)
    if args.mock:
        search_fn, answer_fn = make_mock_backend(questions)
        results_dir = args.results_dir or (config.RESULTS_DIR / "mock")
        sleep = 0.0
    else:
        from src.generation import answer as answer_fn
        from src.retrieval import search as search_fn
        results_dir = args.results_dir or config.RESULTS_DIR
        sleep = args.sleep
    thresholds = ([float(t) for t in args.thresholds.split(",") if t.strip()]
                  if args.thresholds else None)

    rows, metrics = run_eval(questions, args.mode, search_fn, answer_fn,
                             retrieval_only=args.retrieval_only, sleep=sleep,
                             retries=args.retries, retry_wait=args.retry_wait,
                             thresholds=thresholds, verbose=True)
    summary = save_outputs(results_dir, args.mode, rows, metrics, args.retrieval_only)

    print(json.dumps({k: v for k, v in metrics.items() if k != "threshold_sweep"},
                     ensure_ascii=False, indent=2))
    if args.retrieval_only:
        print("\nVarredura de limiares (recusa se melhor similaridade vetorial < limiar):")
        print(render_sweep(metrics["threshold_sweep"]))
    print(f"\nResultados em {results_dir}\n")
    print(render_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
