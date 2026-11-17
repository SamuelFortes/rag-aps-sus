"""Gera tests/fixtures/index/faiss.index a partir de tests/fixtures/index/chunks.jsonl.

Mesmo formato do contrato: config.EMBEDDING_MODEL com PASSAGE_PREFIX, embeddings
normalizados, faiss.IndexFlatIP, posição i = linha i. Uso:
    python -m tests.fixtures.build_fixture_index
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config

INDEX_DIR = Path(__file__).resolve().parent / "index"


def main() -> None:
    with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    emb = model.encode([config.PASSAGE_PREFIX + c["text"] for c in chunks], normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    print(f"{index.ntotal} vetores de dimensão {emb.shape[1]} em {INDEX_DIR / 'faiss.index'}")


if __name__ == "__main__":
    main()
