"""Dev 1 (T1): data/processed/pages.jsonl -> data/index/chunks.jsonl.

Formato de saída por linha: {"id": "<doc>:<page>:<n>", "doc", "page", "text"}.
Regra: um chunk nunca atravessa páginas.
"""


def chunk_page(doc: str, page: int, text: str) -> list[dict]:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
