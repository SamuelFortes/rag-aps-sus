"""Dev 1 (T1): data/index/chunks.jsonl -> data/index/faiss.index.

Embeddings: config.EMBEDDING_MODEL com config.PASSAGE_PREFIX, normalizados,
em faiss.IndexFlatIP. A posição i no índice corresponde à linha i de chunks.jsonl.
Executar com: python -m src.build_index
"""
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_FILE, EMBEDDING_MODEL, FAISS_FILE, PASSAGE_PREFIX


def load_chunks() -> list[dict]:
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    chunks = load_chunks()
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [PASSAGE_PREFIX + c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    FAISS_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_FILE))
    print(f"{index.ntotal} vetores (dim {index.d}) gravados em {FAISS_FILE}")


if __name__ == "__main__":
    main()
