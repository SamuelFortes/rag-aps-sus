import pymupdf

from src.config import RAW_DIR, PAGES_FILE as OUT_FILE

def extract_pages(pdf_path):
    pdf = pymupdf.open(pdf_path)
    pages = []
    for index, page in enumerate(pdf, start=1):
        pages.append(index, page.get_text()) # atrelando a pagina ao texto específico para facilitar na hora de saber a qual página pertence aquele texto
    return pages

def find_repeated_lines(pages):
    conta em quantas páginas cada linha aparece
    repeated_lines = {}
    for page_index, page in enumerate(pages):
        lines = page[page_index].splitlines()
        for line_index, line in enumerate(lines):
            for index, page_after_current_page in enumerate(pages[page_index + 1:]):
                if line[line_index] == page_after_current_page[index][1]:
                    repeated_lines[line[line_index]].append(page_index)






    se aparece em mais de ~30% das páginas, é cabeçalho/rodapé
    devolve o conjunto dessas linhas

função limpar(texto, linhas_repetidas):
    remove as linhas repetidas
    junta palavras hifenizadas no fim da linha (regex: "-\n" entre letras)
    normaliza espaços e quebras de linha
    devolve texto limpo

principal:
    para cada PDF em RAW_DIR:
        páginas = extrair_paginas(pdf)
        repetidas = achar_linhas_repetidas(páginas)
        para cada página:
            texto = limpar(...)
            se len(texto) < ~50 caracteres: pula (capa, página em branco)
            grava uma linha JSON: {"doc": nome do arquivo, "page": n, "text": texto}
