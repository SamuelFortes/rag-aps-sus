# PRD — Assistente RAG para Atenção Primária (SUS)

> Tipo: Produto inicial · Data: 2026-11-17
> Status: Pronto para planejamento

## Visão geral e objetivo

Profissionais da Atenção Primária à Saúde (APS) perdem tempo procurando orientações em documentos extensos do Ministério da Saúde. O produto é um assistente de perguntas e respostas sobre um corpus fixo de documentos oficiais da APS que:

- responde em português citando documento e página de cada afirmação;
- recusa quando a resposta não está no corpus;
- tem a qualidade medida por um conjunto de avaliação, com um baseline e uma melhoria comparados numericamente.

O projeto é um portfólio público (GitHub + demo no Hugging Face Spaces) para o Programa de Estágio GenAI do iFood, trilha AI Engineer. A entrega precisa produzir números verificáveis de qualidade, não apenas uma demo.

## Contexto do projeto

- Repositório em estágio inicial: descrição do projeto (`projeto-rag-aps-sus.md`), dependências fixadas em `requirements.txt` e um script de ingestão incompleto (`src/ingest.py`, parte em pseudocódigo).
- Corpus já baixado em `data/raw`: 5 PDFs públicos do Ministério da Saúde.
- A chave da API do Gemini é lida de variável de ambiente (`.env` local, secret no Spaces).
- Prazo: inscrição até 17/11/2026. Se faltar tempo, a ordem de corte é: (1) a melhoria, depois (2) a interface. A avaliação do baseline é a última coisa a cortar.

## Decisões do produto

- **Corpus fixo** nos 5 PDFs presentes: Carteira de Serviços da APS, PNAB, CAB 32 (Pré-natal), CAB 36 (Diabetes) e CAB 37 (Hipertensão). Só material público.
- **Ingestão** finaliza a lógica já esboçada pelo autor: extração por página, remoção de linhas repetidas em mais de ~30% das páginas (cabeçalhos e rodapés), junção de palavras hifenizadas na quebra de linha, normalização de espaços e descarte de páginas com menos de ~50 caracteres.
- **Chunking dentro da página.** Um chunk nunca atravessa páginas. Chunks de ~800–1000 caracteres com ~150 de sobreposição. Assim a citação "documento + página" é exata.
- **Busca baseline** vetorial com embeddings `intfloat/multilingual-e5-base` e índice FAISS.
- **Melhoria** com busca híbrida BM25 + vetorial, combinada por Reciprocal Rank Fusion (RRF), medida contra o baseline.
- **Geração** com Gemini (`gemini-2.5-flash` por padrão, configurável por variável de ambiente) em temperatura 0. O modelo recebe os 5 chunks mais relevantes, cada um identificado por documento e página.
- **Citação inline** no formato `[documento, p. N]`. A interface também lista as fontes usadas.
- **Recusa em duas camadas.** (1) Se a similaridade do melhor resultado ficar abaixo de um limiar calibrado no conjunto de avaliação, o sistema recusa sem chamar o LLM. (2) O prompt instrui o Gemini a recusar quando o contexto não responde. A recusa é sempre o texto fixo: "Não encontrei essa informação nos documentos disponíveis."
- **Índice versionado.** Chunks e índice FAISS são gerados localmente e commitados. A demo apenas os carrega, sem reprocessar os PDFs na inicialização.
- **Conjunto de avaliação** com ~35 perguntas (≈30 respondíveis pelo corpus e ≈5 fora do escopo), cada uma com documento e página corretos anotados (uma ou mais páginas válidas). O rascunho é gerado com IA a partir do corpus e **revisado e aprovado pelo autor** antes de qualquer medição.
- **Demo** em Gradio com uma pergunta por vez, sem histórico. O autor publica no Hugging Face Spaces com a própria conta.
- **Idioma.** Respostas, interface, README e documentação em português.
- **Aviso** na interface de que o assistente é um projeto de demonstração e não substitui orientação profissional nem os documentos oficiais.

## Histórias de usuário

- Como profissional da APS, quero fazer uma pergunta em português e receber uma resposta baseada nos documentos oficiais, para não ter que procurar manualmente em PDFs extensos.
- Como profissional da APS, quero ver documento e página de cada afirmação, para conferir a informação na fonte.
- Como profissional da APS, quero que o assistente diga claramente quando não encontrou a resposta, para não agir com base em informação inventada.
- Como autor do projeto, quero rodar a ingestão e gerar o índice a partir dos PDFs, para reconstruir a base quando o corpus ou o chunking mudar.
- Como autor do projeto, quero rodar a avaliação e obter hit@5, citação correta e recusa correta para o baseline e para a busca híbrida, para publicar a comparação no README.
- Como autor do projeto, quero revisar o conjunto de perguntas antes da medição, para que os números sejam confiáveis.
- Como recrutador ou avaliador, quero abrir uma demo pública e ler um README com resultados, para verificar o projeto sem instalar nada.

## Regras de negócio e dados

**Dados conceituais**
- *Página processada:* documento (nome do arquivo), número da página (base 1, igual à numeração do PDF) e texto limpo.
- *Chunk:* identificador, documento, página e texto. Pertence a exatamente uma página.
- *Pergunta de avaliação:* texto da pergunta, tipo (no escopo / fora do escopo) e, se no escopo, o documento e a lista de páginas válidas.
- *Resultado de avaliação:* configuração de busca (baseline ou híbrida), métricas agregadas e o resultado por pergunta.

**Regras**
- Toda resposta não recusada contém ao menos uma citação `[documento, p. N]`, e cada citação aponta para um chunk que foi efetivamente enviado ao modelo.
- Toda recusa usa exatamente o texto fixo de recusa.
- O limiar de similaridade é um parâmetro configurável, calibrado nas perguntas de avaliação.
- O nome do modelo Gemini e a chave da API vêm de variáveis de ambiente. A chave nunca é versionada.
- Sem chave de API, ou com erro do Gemini, a interface mostra uma mensagem de erro amigável em vez de travar.

**Métricas**
- *hit@5:* fração das perguntas no escopo em que algum dos 5 primeiros chunks vem do documento e de uma das páginas anotadas.
- *Citação correta:* fração das perguntas no escopo em que a resposta cita o documento e uma das páginas anotadas.
- *Recusa correta:* fração das perguntas fora do escopo em que o sistema responde com o texto fixo de recusa. Também se registra quantas perguntas no escopo foram recusadas indevidamente.

## Fluxos

### Construção da base

1. O autor executa a ingestão sobre `data/raw`.
2. O sistema extrai o texto por página, limpa cabeçalhos, rodapés e hifenização, descarta páginas quase vazias e grava as páginas processadas.
3. O sistema divide cada página em chunks, gera os embeddings, monta o índice vetorial (e o BM25 para a busca híbrida) e salva os artefatos.
4. Se um PDF não puder ser lido, o sistema informa qual arquivo falhou e segue com os demais.

### Pergunta na demo

1. O usuário digita uma pergunta ou clica numa pergunta de exemplo.
2. O sistema recupera os 5 chunks mais relevantes.
3. Se a similaridade do melhor chunk estiver abaixo do limiar, o sistema exibe a recusa sem chamar o LLM.
4. Caso contrário, o Gemini gera a resposta com citações inline, ou recusa com o texto fixo se o contexto não responder.
5. A interface mostra a resposta e, abaixo, as fontes usadas (documento, página e trecho).
6. Em erro de API, o sistema mostra uma mensagem amigável.

### Avaliação

1. O autor gera o rascunho das perguntas a partir do corpus, revisa, corrige e aprova.
2. O autor executa a avaliação para o baseline e para a busca híbrida.
3. O sistema calcula as três métricas por configuração, grava os resultados por pergunta e agregados, e produz a tabela comparativa para o README.
4. O autor usa as perguntas de avaliação para calibrar o limiar de recusa.

### Publicação

1. O autor publica o app no Hugging Face Spaces e cadastra o secret `GEMINI_API_KEY`.
2. O README traz a descrição, como rodar, a tabela de resultados (baseline × híbrida), o link da demo e um GIF.

## Critérios de aceite

- Dado o corpus em `data/raw`, quando a ingestão roda, então são geradas páginas processadas com documento, página e texto, sem as linhas repetidas de cabeçalho e rodapé e sem páginas com menos de ~50 caracteres.
- Dado qualquer chunk gerado, quando se confere sua origem, então todo o seu texto pertence a uma única página do documento indicado.
- Dados os artefatos de índice versionados, quando a demo inicia, então ela responde sem reprocessar os PDFs.
- Dada uma pergunta coberta pelo corpus, quando o usuário pergunta, então a resposta está em português e contém ao menos uma citação `[documento, p. N]` que corresponde a um chunk recuperado.
- Dada uma pergunta fora do corpus, quando o usuário pergunta, então a resposta é exatamente "Não encontrei essa informação nos documentos disponíveis."
- Dada a melhor similaridade abaixo do limiar, quando o usuário pergunta, então o sistema recusa sem chamar o Gemini.
- Dada a ausência da chave de API, quando o usuário pergunta, então a interface exibe uma mensagem de erro em vez de travar.
- Dado o conjunto de avaliação aprovado, quando a avaliação roda, então são produzidos hit@5, citação correta e recusa correta para o baseline e para a busca híbrida, junto com os resultados por pergunta.
- Dados os resultados da avaliação, quando se abre o README, então ele contém a tabela comparativa baseline × híbrida com os números medidos.
- Dada a demo aberta, quando o usuário vê a interface, então ela mostra perguntas de exemplo, as fontes da resposta e o aviso de que não substitui orientação profissional.

## Stack e restrições técnicas decididas

- **Python** com as dependências já fixadas em `requirements.txt`.
- **pymupdf** para extração de texto por página.
- **sentence-transformers** com `intfloat/multilingual-e5-base`: bom em português e roda em CPU no Spaces gratuito.
- **FAISS (faiss-cpu)**: suficiente para o tamanho do corpus e fácil de embarcar no Spaces.
- **rank-bm25** para a componente lexical da busca híbrida.
- **google-genai** com Gemini `gemini-2.5-flash`, por ter free tier e baixa latência.
- **python-dotenv** para configuração local.
- **Gradio** para a interface e **Hugging Face Spaces** (CPU gratuito) para hospedagem.
