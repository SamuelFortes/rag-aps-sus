from src.chunking import chunk_page, split_text
from src.config import CHUNK_SIZE


def _long_text(n_sentences: int, tag: str) -> str:
    return " ".join(f"Frase número {i} da página {tag} sobre cuidado na atenção primária." for i in range(n_sentences))


def test_nenhum_chunk_atravessa_paginas():
    pages = [("doc.pdf", 1, _long_text(40, "UM")), ("doc.pdf", 2, _long_text(40, "DOIS"))]
    chunks = [c for doc, page, text in pages for c in chunk_page(doc, page, text)]
    assert len(chunks) > 2
    for c in chunks:
        assert c["id"].startswith(f"{c['doc']}:{c['page']}:")
        if c["page"] == 1:
            assert "DOIS" not in c["text"]
        else:
            assert "UM" not in c["text"]


def test_ids_unicos_e_sequenciais():
    chunks = chunk_page("doc.pdf", 7, _long_text(60, "X"))
    assert [c["id"] for c in chunks] == [f"doc.pdf:7:{n}" for n in range(len(chunks))]


def test_tamanho_e_quebra_em_fim_de_frase():
    pieces = split_text(_long_text(60, "X"))
    assert all(len(p) <= CHUNK_SIZE for p in pieces)
    assert all(p.endswith(".") for p in pieces)


def test_sobreposicao_entre_chunks_consecutivos():
    pieces = split_text(_long_text(60, "X"))
    for a, b in zip(pieces, pieces[1:]):
        assert b[:30] in a


def test_pagina_curta_vira_um_chunk():
    assert split_text("Texto curto.") == ["Texto curto."]
    assert split_text("   ") == []


def test_texto_sem_espacos_nao_trava():
    pieces = split_text("a" * (CHUNK_SIZE * 3))
    assert sum(len(p) for p in pieces) >= CHUNK_SIZE * 3
