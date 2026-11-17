"""Dev 1 (T1): PDFs em data/raw -> data/processed/pages.jsonl.

Formato de saída por linha: {"doc": "<arquivo.pdf>", "page": <int, base 1>, "text": "<texto limpo>"}.
Executar com: python -m src.ingest
"""
import json
import re
import sys
from collections import Counter

import pymupdf

from src.config import MIN_PAGE_CHARS, RAW_DIR, REPEATED_LINE_RATIO, PAGES_FILE as OUT_FILE

# palavra hifenizada no fim da linha: letra + "-" + quebra + letra
_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[^\W\d_])")
# hífen que deve ser mantido (só a quebra sai): pronome enclítico/mesoclítico
# ("Deve-\nse" -> "Deve-se") ou linha seguinte começando com maiúscula
_ENCLITICS = r"(?:se|lo|la|los|las|lhe|lhes|me|te|nos|vos|o|a|os|as)\b"
_UPPERCASE = r"[A-ZÀ-ÖØ-Þ]"
_KEEP_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=" + _ENCLITICS + "|" + _UPPERCASE + ")")
# linha que é só número de página, inclusive repetido ("33" ou "33 33 33 33")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*(\d{1,4})(?:\s+\1)*\s*$")
_WHITESPACE_RE = re.compile(r"\s+")
# linhas muito curtas (ex.: incisos "I", "II") não são tratadas como cabeçalho/rodapé
_MIN_HEADER_LEN = 4


def extract_pages(pdf_path) -> list[tuple[int, str]]:
    with pymupdf.open(pdf_path) as pdf:
        pages = []
        for index, page in enumerate(pdf, start=1):
            pages.append((index, page.get_text()))  # atrelando a pagina ao texto específico para facilitar na hora de saber a qual página pertence aquele texto
    return pages


def _line_key(line: str) -> str:
    return _WHITESPACE_RE.sub(" ", line).strip()


def find_repeated_lines(pages: list[tuple[int, str]], ratio: float = REPEATED_LINE_RATIO) -> set[str]:
    # conta em quantas páginas cada linha aparece (cada linha conta no máximo uma vez por página)
    page_count = Counter()
    for _, text in pages:
        lines = {_line_key(line) for line in text.splitlines()}
        lines.discard("")
        page_count.update(lines)

    # se aparece em mais de ~30% das páginas, é cabeçalho/rodapé
    # devolve o conjunto dessas linhas
    if not pages:
        return set()
    total = len(pages)
    # (só conta como repetida se aparece em pelo menos 2 páginas, senão em PDFs curtos tudo seria removido)
    return {
        line
        for line, n in page_count.items()
        if n >= 2 and n / total > ratio and len(line) >= _MIN_HEADER_LEN
    }


def clean_text(text: str, repeated_lines: set[str]) -> str:
    # remove as linhas repetidas
    lines = [line for line in text.splitlines() if _line_key(line) not in repeated_lines]
    # remove linhas que são só número de página
    lines = [line for line in lines if not _PAGE_NUMBER_LINE_RE.match(line)]
    text = "\n".join(lines)
    # mantém o hífen (sem a quebra) em ênclise/mesóclise e antes de maiúscula
    text = _KEEP_HYPHEN_RE.sub("-", text)
    # junta palavras hifenizadas no fim da linha (regex: "-\n" entre letras)
    text = _HYPHEN_RE.sub("", text)
    # normaliza espaços e quebras de linha
    return _WHITESPACE_RE.sub(" ", text).strip()


def process_pdf(pdf_path) -> list[dict]:
    pages = extract_pages(pdf_path)
    repeated = find_repeated_lines(pages)
    records = []
    for page_number, raw_text in pages:
        text = clean_text(raw_text, repeated)
        if len(text) < MIN_PAGE_CHARS:  # pula (capa, página em branco)
            continue
        records.append({"doc": pdf_path.name, "page": page_number, "text": text})
    return records


def main() -> None:
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Nenhum PDF encontrado em {RAW_DIR}", file=sys.stderr)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8", newline="\n") as out:
        for pdf_path in pdfs:
            try:
                records = process_pdf(pdf_path)
            except Exception as exc:  # PDF ilegível: avisa e segue
                print(f"AVISO: não foi possível ler {pdf_path.name}: {exc}", file=sys.stderr)
                continue
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{pdf_path.name}: {len(records)} páginas")
    print(f"Gravado em {OUT_FILE}")


if __name__ == "__main__":
    main()
