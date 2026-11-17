"""Configuração central do projeto. Caminhos relativos à raiz do repositório."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Caminhos
RAW_DIR = ROOT / "data" / "raw"              # PDFs (ignorado pelo git)
PROCESSED_DIR = ROOT / "data" / "processed"  # ignorado pelo git
INDEX_DIR = ROOT / "data" / "index"          # versionado
PAGES_FILE = PROCESSED_DIR / "pages.jsonl"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
FAISS_FILE = INDEX_DIR / "faiss.index"
EVAL_DIR = ROOT / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# Ingestão
REPEATED_LINE_RATIO = 0.30   # linha em > 30% das páginas = cabeçalho/rodapé
MIN_PAGE_CHARS = 50          # páginas com menos que isso são descartadas

# Chunking (nunca atravessa páginas)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Embeddings
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Busca
TOP_K = 5
RRF_K = 60
SEARCH_MODES = ("vector", "hybrid")
DEFAULT_MODE = "hybrid"

# Recusa
SIMILARITY_THRESHOLD = 0.80  # provisório; calibrado na T4
RECUSA = "Não encontrei essa informação nos documentos disponíveis."

# Geração
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE = 0.0
