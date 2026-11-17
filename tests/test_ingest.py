from src.config import MIN_PAGE_CHARS
from src.ingest import clean_text, find_repeated_lines, process_pdf

HEADER = "Ministério da Saúde | Secretaria de Atenção à Saúde"


def test_hifenizacao_no_fim_da_linha_e_juntada():
    text = "A gestante deve realizar acompa-\nnhamento regular.\nOutra linha."
    assert clean_text(text, set()) == "A gestante deve realizar acompanhamento regular. Outra linha."


def test_hifen_entre_numeros_nao_e_juntado():
    assert clean_text("entre 10-\n20 semanas", set()) == "entre 10- 20 semanas"


def test_hifen_mantido_antes_de_pronome_encliticos():
    assert clean_text("Deve-\nse evitar.", set()) == "Deve-se evitar."
    assert clean_text("Trata-\nse de", set()) == "Trata-se de"
    assert clean_text("orientá-\nla sobre", set()) == "orientá-la sobre"
    assert clean_text("realizar-\nse-á", set()) == "realizar-se-á"
    assert clean_text("encaminhá-\nlos.", set()) == "encaminhá-los."
    # palavra comum que só começa como pronome continua sendo juntada
    assert clean_text("pres-\nsão arterial", set()) == "pressão arterial"
    assert clean_text("acompa-\nnhamento", set()) == "acompanhamento"
    assert clean_text("diag-\nnóstico", set()) == "diagnóstico"


def test_hifen_mantido_antes_de_maiuscula():
    assert clean_text("Ministério da Saúde-\nBrasil", set()) == "Ministério da Saúde-Brasil"
    assert clean_text("pré-\nÉclampsia", set()) == "pré-Éclampsia"


def test_remove_linhas_de_numero_de_pagina():
    text = "33\nConteúdo da página.\n 170 \n33 33 33 33\nTotal de 6 consultas.\n12345"
    assert clean_text(text, set()) == "Conteúdo da página. Total de 6 consultas. 12345"
    # números diferentes na mesma linha (ex.: tabela) não são removidos
    assert clean_text("10 20 30", set()) == "10 20 30"


def test_normaliza_espacos_e_quebras():
    assert clean_text("  texto   com\n\n\tespaços  \n", set()) == "texto com espaços"


def test_cabecalho_repetido_e_removido():
    pages = [(i, f"{HEADER}\n{i}\nConteúdo único da página {i} sobre pré-natal.") for i in range(1, 11)]
    repeated = find_repeated_lines(pages)
    assert HEADER in repeated
    assert not any("Conteúdo único" in line for line in repeated)
    cleaned = clean_text(pages[0][1], repeated)
    assert HEADER not in cleaned
    assert "Conteúdo único da página 1" in cleaned


def test_linha_em_poucas_paginas_nao_e_cabecalho():
    pages = [(i, f"Texto {i} " * 3) for i in range(1, 11)]
    pages[0] = (1, "Linha rara\n" + pages[0][1])
    pages[1] = (2, "Linha rara\n" + pages[1][1])  # 20% das páginas, abaixo de 30%
    assert "Linha rara" not in find_repeated_lines(pages)


def test_pagina_curta_e_descartada(tmp_path):
    import pymupdf

    pdf_path = tmp_path / "teste.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Capa")  # menos que MIN_PAGE_CHARS
    long_text = "Texto suficientemente longo\npara ser mantido\nno índice da página."
    assert len(long_text) >= MIN_PAGE_CHARS
    doc.new_page().insert_text((72, 72), long_text)
    doc.save(pdf_path)
    doc.close()

    records = process_pdf(pdf_path)
    assert [r["page"] for r in records] == [2]
    assert records[0]["doc"] == "teste.pdf"
