# Assistente RAG para Atenção Primária (SUS)

Projeto pessoal para o Programa de Estágio GenAI do iFood (trilha AI Engineer).

## Motivação

No PIBITI, trabalhando com gestão de Unidades Básicas de Saúde na plataforma MeuTerritório, percebi que profissionais da Atenção Primária perdem tempo procurando orientações em documentos extensos do Ministério da Saúde. Este projeto testa se um assistente com IA consegue responder essas dúvidas de forma confiável, citando a fonte, e mede isso.

É um projeto independente, motivado pela experiência na IC, não uma continuação do MeuTerritório.

## Objetivos

- Ter um projeto de IA público e verificável, com repositório no GitHub e demo no Hugging Face Spaces.
- Mostrar métricas de qualidade de um sistema com LLM.
- Praticar retrieval em Python: embeddings, busca vetorial e RAG.

## Escopo

### Corpus

5 a 10 PDFs públicos do Ministério da Saúde, por exemplo:

- Carteira de Serviços da Atenção Primária à Saúde
- Política Nacional de Atenção Básica (PNAB)
- Cadernos de Atenção Básica selecionados

Apenas material público.

### Pipeline

1. Extração de texto dos PDFs, guardando documento e página.
2. Chunking.
3. Embeddings com modelo multilíngue aberto do Hugging Face (sentence-transformers).
4. Índice vetorial em FAISS ou Chroma.
5. Geração com Gemini, com duas regras:
   - toda resposta cita documento e página;
   - se a resposta não estiver no corpus, o sistema recusa.

### Avaliação

Conjunto de 30 a 40 perguntas com a fonte correta anotada, incluindo cerca de 5 perguntas fora do escopo.

| Métrica | O que mede |
|---|---|
| hit@5 | O trecho correto aparece entre os 5 primeiros resultados da busca |
| Citação correta | A resposta cita o documento e a página certos |
| Recusa correta | O sistema recusa nas perguntas fora do escopo |

Fluxo: medir o baseline, implementar uma melhoria (busca híbrida BM25 + vetorial, ou reranker) e medir de novo. O antes e depois vai numa tabela no README.

### Demo

Interface em Gradio publicada no Hugging Face Spaces.

## Stack

Python, Gemini, sentence-transformers, FAISS ou Chroma, rank-bm25, Gradio, Hugging Face Spaces.

## Cronograma

Prazo de inscrição: 17/11/2026. Meta de envio: 15/11.

| Dias | Entrega |
|---|---|
| 4–5/11 | Baixar o corpus, extração e chunking, montar o índice |
| 6–7/11 | Geração com citação e recusa, primeira versão funcionando |
| 8–9/11 | Escrever as perguntas e o script de avaliação, medir o baseline |
| 10–11/11 | Implementar uma melhoria e medir de novo |
| 12–13/11 | Deploy no Spaces, README com tabela de resultados e GIF |
| 14–15/11 | Atualizar o currículo e enviar |

Se faltar tempo, cortar nesta ordem:

1. A melhoria (ficar só com o baseline medido).
2. A interface (uma demo em notebook já serve).
3. A avaliação é a última coisa a cortar, porque é ela que gera os números.

## Entregáveis

- [ ] Repositório público no GitHub com README
- [ ] Tabela de resultados da avaliação (baseline e melhoria)
- [ ] Demo no Hugging Face Spaces
- [ ] GIF ou vídeo curto da demo
- [ ] Projeto adicionado ao currículo

## Como entra no currículo

**Assistente RAG para Atenção Primária (SUS)** | Python, Gemini, sentence-transformers, FAISS, Hugging Face Spaces

- RAG sobre documentos oficiais da APS com citação de fonte e página e recusa quando a resposta não está no corpus.
- Conjunto de avaliação com [N] perguntas: hit@5 de [X]% → [Y]% com busca híbrida; [Z]% de recusa correta fora do escopo.
- Demo pública no Hugging Face Spaces e código aberto no GitHub.
