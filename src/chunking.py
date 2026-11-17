"""Dev 1 (T1): data/processed/pages.jsonl -> data/index/chunks.jsonl.

Formato de saída por linha: {"id": "<doc>:<page>:<n>", "doc", "page", "text"}.
Regra: um chunk nunca atravessa páginas.
Executar com: python -m src.chunking
"""
import json
import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_FILE, PAGES_FILE

# fim de frase: pontuação final seguida de espaço
_SENTENCE_END_RE = re.compile(r"[.!?;:](?=\s)")
# o corte em fim de frase só é aceito na segunda metade da janela, para não gerar chunks curtos
_MIN_CUT_FRACTION = 0.5


def _find_cut(text: str, start: int, end: int) -> int:
    """Posição de corte em (start, end]: fim de frase, senão espaço, senão `end`."""
    window_start = start + int((end - start) * _MIN_CUT_FRACTION)
    best = -1
    for match in _SENTENCE_END_RE.finditer(text, window_start, end):
        best = match.end()
    if best > 0:
        return best
    space = text.rfind(" ", window_start, end)
    return space if space > start else end


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    pieces = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            pieces.append(text[start:].strip())
            break
        cut = _find_cut(text, start, end)
        pieces.append(text[start:cut].strip())
        # próximo chunk recomeça `overlap` caracteres antes do corte, alinhado ao início de uma palavra
        next_start = max(cut - overlap, start + 1)
        space = text.find(" ", next_start, cut)
        if space != -1:
            next_start = space + 1
        start = next_start
    return [p for p in pieces if p]


def chunk_page(doc: str, page: int, text: str) -> list[dict]:
    return [
        {"id": f"{doc}:{page}:{n}", "doc": doc, "page": page, "text": piece}
        for n, piece in enumerate(split_text(text))
    ]


def main() -> None:
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(PAGES_FILE, encoding="utf-8") as src, open(CHUNKS_FILE, "w", encoding="utf-8", newline="\n") as out:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            for chunk in chunk_page(record["doc"], record["page"], record["text"]):
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                counts[chunk["doc"]] = counts.get(chunk["doc"], 0) + 1
    for doc, n in counts.items():
        print(f"{doc}: {n} chunks")
    print(f"Total: {sum(counts.values())} chunks gravados em {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
