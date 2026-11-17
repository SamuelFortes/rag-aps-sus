# Plano de implementação — Assistente RAG para Atenção Primária (SUS)

> Fonte: [`docs/prd/assistente-rag-aps-sus.md`](../prd/assistente-rag-aps-sus.md) · Data: 2026-11-17
> Execução: multi-agentes via Orca (Run + DAG de tarefas)
> Status: Aguardando aprovação

## 1. Organização dos agentes

| Papel | Quem | Responsabilidades |
|---|---|---|
| **Gerente** | Sessão principal do Claude Code (coordenador Orca) | Cria a Run e o DAG, executa a T0, dispara os workers, revisa cada entrega, faz o merge em `master`, conduz os portões com o usuário e reporta. Não implementa as features. |
| **Dev 1 — Dados** | Worker Orca `claude`, worktree filha `dev-dados` | Ingestão, chunking, embeddings, índice FAISS e artefatos versionados. |
| **Dev 2 — RAG** | Worker Orca `claude`, worktree filha `dev-rag` | Busca vetorial e híbrida, geração com Gemini, citação, recusa e interface Gradio. |
| **Dev 3 — Avaliação** | Worker Orca `claude`, worktree filha `dev-eval` | Script de métricas, rascunho das perguntas, medição, calibração do limiar e README. |

**Regras gerais**
- Cada dev trabalha **apenas** na própria worktree filha e branch. Commits são locais e ninguém dá `push`.
- Ao terminar uma tarefa, o dev faz commit e envia `worker_done` com evidências (comandos rodados e saída).
- O gerente revisa o diff, roda as verificações da tarefa, faz o merge em `master` e libera as próximas tarefas.
- Correções vão para o próprio dev (via `orca orchestration send`). O gerente não corrige o código dos devs.
- Perguntas de produto que não estão resolvidas no PRD sobem para o gerente (`ask`), que consulta o usuário quando necessário.

**Restrições de ambiente que todo worker deve conhecer**
- `data/raw/` (PDFs), `data/processed/` e `.env` estão no `.gitignore`, então **não existem nas worktrees filhas**. O **primeiro passo de todo worker**, antes de qualquer código, é ligá-los ao checkout principal (sem cópia e sem admin):

  ```powershell
  $main = "C:\Users\077114263\Documents\projeto-RAG"
  New-Item -ItemType Directory -Force data | Out-Null
  if (-not (Test-Path data\raw))       { cmd /c mklink /J data\raw       "$main\data\raw" }
  if (-not (Test-Path data\processed)) { New-Item -ItemType Directory -Force "$main\data\processed" | Out-Null; cmd /c mklink /J data\processed "$main\data\processed" }
  if (-not (Test-Path .env))           { cmd /c mklink /H .env           "$main\.env" }
  ```

  Assim o código usa caminhos relativos normais (`data/raw`, `data/processed`, `load_dotenv()`), igual no checkout principal e no Spaces.
- **Proibições sobre as ligações:** elas apontam para os arquivos reais.
  - Nunca escrever, mover ou apagar em `data/raw` nem no `.env`, e nunca apagar as próprias ligações com remoção recursiva.
  - Nunca imprimir, ler para o relatório nem commitar o `.env`.
  - **Só o Dev 1 escreve em `data/processed`.** Os demais apenas leem.
- Python: usar o venv do checkout principal (`C:\Users\077114263\Documents\projeto-RAG\.venv\Scripts\python.exe`). Só a T0 instala dependências. Nenhum worker roda `pip install` sem pedir ao gerente.
- Os artefatos versionados ficam em `data/index/`, que não está ignorado. `data/processed/` continua ignorado.

## 2. Contrato de dados e interfaces (definido na T0)

Esse contrato permite que Dev 2 e Dev 3 trabalhem em paralelo ao Dev 1.

```text
src/
  config.py      # caminhos, nomes de modelo, TOP_K=5, CHUNK_SIZE, OVERLAP, limiar, texto de recusa
  ingest.py      # Dev 1: PDFs -> data/processed/pages.jsonl
  chunking.py    # Dev 1: pages.jsonl -> data/index/chunks.jsonl
  build_index.py # Dev 1: chunks -> data/index/faiss.index (+ metadados)
  retrieval.py   # Dev 2: search(query, k, mode="vector"|"hybrid") -> list[Hit]
  generation.py  # Dev 2: answer(question, mode) -> Answer
  app.py         # Dev 2: Gradio
eval/
  questions.jsonl  # Dev 3 (rascunho -> aprovado pelo usuário)
  run_eval.py      # Dev 3
  results/         # saída da avaliação
```

**Formatos**
- `pages.jsonl`: `{"doc": "<arquivo.pdf>", "page": <int, base 1>, "text": "<texto limpo>"}`
- `chunks.jsonl`: `{"id": "<doc>:<page>:<n>", "doc": ..., "page": ..., "text": ...}`. Cada chunk pertence a uma única página.
- `Hit`: `id, doc, page, text, score`. O score vetorial é a similaridade de cosseno, com o prefixo `query: `/`passage: ` exigido pelo e5.
- `Answer`: `text, refused: bool, citations: list[(doc, page)], sources: list[Hit], error: str|None`
- Citação no texto: `[<doc sem .pdf>, p. N]`. Recusa: exatamente `RECUSA = "Não encontrei essa informação nos documentos disponíveis."`
- `questions.jsonl`: `{"id", "question", "in_scope": bool, "doc": str|null, "pages": [int]}`

## 3. DAG de tarefas

```text
T0 (gerente) ──┬─> T1 (Dev1) ──┬──────────────> T3b (Dev3) ─> G1 (usuário) ─┐
               │               │                                           │
               ├─> T2 (Dev2) ──┼─> T5 (Dev2: Gradio) ──────────────────────┼─> T6 (Dev3: README) ─> G2 (usuário: deploy)
               │               │                                           │
               └─> T3a (Dev3) ─┴───────────────────────────> T4 (Dev3: medição) ┘
```

| ID | Responsável | Depende de | Entrega |
|---|---|---|---|
| T0 | Gerente | — | Contrato e esqueleto |
| T1 | Dev 1 | T0 | Ingestão, chunking e índice |
| T2 | Dev 2 | T0 | Busca, geração e recusa |
| T3a | Dev 3 | T0 | Script de avaliação |
| T3b | Dev 3 | T1 | Rascunho das perguntas |
| G1 | Usuário | T3b | Aprovação das perguntas |
| T4 | Dev 3 | T1, T2, T3a, G1 | Medição baseline × híbrida e calibração do limiar |
| T5 | Dev 2 | T1, T2 | App Gradio pronto para o Spaces |
| T6 | Dev 3 | T4, T5 | README com resultados |
| G2 | Usuário | T6 | Deploy no Spaces e GIF |

Em paralelo após a T0: T1, T2 e T3a. Depois da T1: T3b e, com a T2, a T5.

**Prioridade em caso de falta de tempo**, seguindo o PRD: T4 só com o baseline, se a parte híbrida atrasar, e a T5 pode ser cortada. A T4 (baseline) é a última a cortar.

## 4. Especificação das tarefas

### T0 — Contrato e esqueleto (gerente)
- Instalar `requirements.txt` no `.venv` principal e conferir os imports.
- Criar `src/config.py` com as constantes da seção 2. Os caminhos são relativos à raiz do repositório (`data/raw`, `data/processed`, `data/index`), resolvidos a partir da localização do próprio `config.py`. O `.env` é carregado com `load_dotenv()`, e `GEMINI_API_KEY` e `GEMINI_MODEL` (padrão `gemini-3.5-flash-lite`; o `gemini-2.5-flash` original não está disponível para a chave do projeto) vêm do ambiente.
- Substituir os caminhos absolutos do `src/ingest.py` atual pelos de `config.py`, sem mexer na lógica esboçada, que fica para a T1.
- Criar os módulos vazios com assinaturas e docstrings do contrato, além de `eval/` e `data/index/`.
- Commit em `master`, que é a base das worktrees filhas.
- **Pronto quando** `python -c "import src.config"` roda sem erro.

### T1 — Ingestão, chunking e índice (Dev 1)
- Terminar o `src/ingest.py` mantendo a lógica esboçada pelo autor: extração por página com pymupdf, linhas repetidas em mais de ~30% das páginas removidas, hifenização no fim da linha juntada, espaços normalizados, páginas com menos de ~50 caracteres descartadas. Um PDF ilegível gera aviso e o processo continua.
- `chunking.py`: chunks de ~800–1000 caracteres com ~150 de sobreposição, **sem atravessar páginas**, quebrando preferencialmente em fim de frase.
- `build_index.py`: embeddings `intfloat/multilingual-e5-base` com prefixo `passage: `, normalizados, em um FAISS `IndexFlatIP`. Salvar em `data/index/`.
- Testes simples (pytest): limpeza de hifenização e de cabeçalho, e nenhum chunk com mais de uma página.
- **Pronto quando** os três scripts rodam do zero sobre os 5 PDFs, `data/index/` está commitado, o relatório traz a contagem de páginas e chunks por documento e 3 exemplos de chunks limpos, e os testes passam.

### T2 — Busca, geração e recusa (Dev 2)
- `retrieval.py`: modo `vector` (FAISS, prefixo `query: `) e modo `hybrid` (BM25 sobre os chunks, com tokenização simples em minúsculas e sem acentos, combinado com o vetorial por RRF com k=60). Retorna top-5 `Hit`.
- `generation.py`: se o melhor score vetorial for menor que o `LIMIAR` (valor inicial provisório em `config.py`), recusa sem chamar o LLM. Caso contrário, monta o prompt com os 5 chunks identificados por `[doc, p. N]` e instruções para responder em português, citar inline no formato do contrato e responder exatamente `RECUSA` quando o contexto não responde. Usa temperatura 0. Extrai as citações por regex e descarta citações que não estão entre os chunks enviados.
- Sem chave ou com erro de API, `Answer.error` é preenchido e nada lança exceção para o chamador.
- Enquanto a T1 não estiver mergeada, usar um índice-fixture pequeno criado pelo próprio dev, só para testes.
- **Pronto quando** os testes unitários passam (parse de citação, recusa por limiar com o LLM mockado, RRF) e um teste manual com o índice real, depois do merge da T1, mostra 1 resposta com citação e 1 recusa.

### T3a — Script de avaliação (Dev 3)
- `eval/run_eval.py --mode vector|hybrid`: para cada pergunta, calcula hit@5 (algum hit com `doc` e uma das `pages` anotadas), citação correta (alguma citação da resposta coincide com o doc e uma das páginas) e recusa correta (perguntas fora do escopo com `refused`), e conta as recusas indevidas no escopo.
- Grava `eval/results/<modo>.jsonl` (por pergunta) e `eval/results/summary.json`, e gera a tabela Markdown baseline × híbrida.
- Opção `--retrieval-only` para medir o hit@5 sem chamar o Gemini.
- **Pronto quando** o script roda contra um `questions.jsonl` de exemplo com `answer` mockado e produz a tabela.

### T3b — Rascunho das perguntas (Dev 3)
- A partir de `data/processed/pages.jsonl`, lido pela junction e gerado pelo Dev 1, redigir ~35 perguntas: ~30 respondíveis, distribuídas pelos 5 documentos e com o `doc` e as `pages` corretos conferidos no texto, e ~5 fora do escopo (temas de saúde não cobertos e temas não relacionados).
- Gerar `eval/questions_review.md` legível, com pergunta, fonte anotada e trecho de apoio, para a revisão do usuário.
- **Pronto quando** os dois arquivos estão commitados. A tarefa termina no portão G1.

### G1 — Aprovação das perguntas (usuário)
- O gerente apresenta `eval/questions_review.md` ao usuário. As correções voltam ao Dev 3. A T4 só é liberada com o "aprovado" explícito do usuário.

### T4 — Medição e calibração (Dev 3)
- Calibrar o `LIMIAR`: varrer valores com `--retrieval-only` sobre as perguntas aprovadas e escolher o que maximiza a recusa correta sem recusar indevidamente perguntas no escopo. Registrar a escolha em `config.py`, com a justificativa no commit.
- Rodar a avaliação completa em `vector` (baseline) e em `hybrid`, e commitar os resultados e a tabela.
- **Pronto quando** `summary.json` traz as 3 métricas para as duas configurações e o relatório mostra a tabela e as 3 piores falhas comentadas.

### T5 — App Gradio (Dev 2)
- `app.py` (também exposto como `app.py` na raiz, como o Spaces espera): caixa de pergunta, seletor oculto ou fixo no melhor modo medido na T4 (padrão `hybrid`), resposta com citações, lista de fontes (doc, página e trecho), perguntas de exemplo clicáveis e o aviso "projeto de demonstração, não substitui orientação profissional nem os documentos oficiais". Mostra mensagem amigável em caso de erro.
- Carrega só `data/index/`, sem acessar os PDFs.
- **Pronto quando** `python app.py` sobe localmente, e o relatório traz um screenshot ou a saída de uma pergunta respondida e uma recusada, além das instruções de deploy (secret `GEMINI_API_KEY`).

### T6 — README (Dev 3)
- README em português com o problema, a arquitetura em alto nível, como rodar (ingestão → índice → app → avaliação), a tabela de resultados da T4, a metodologia da avaliação, as limitações, um espaço para o link da demo e o GIF, e um resumo curto em inglês.
- **Pronto quando** o README está commitado e todos os números conferem com `summary.json`.

### G2 — Publicação (usuário)
- O gerente entrega o checklist: criar o Space Gradio, enviar o repositório, cadastrar o secret `GEMINI_API_KEY`, gravar o GIF, preencher o link no README e dar `push`.

## 5. Execução no Orca (gerente)

```text
orca orchestration run-create --objective "Assistente RAG APS/SUS conforme docs/prd/assistente-rag-aps-sus.md"
orca orchestration task-create --spec "<spec T1>" --deps '["<T0>"]'   # idem para as demais, com as deps da seção 3
orca orchestration task-list --ready --brief
orca orchestration worker-start --task <id> --agent claude --worktree new-child --name dev-dados   # dev-rag, dev-eval
```

- Cada spec de worker começa com o bloco de ligações e as proibições da seção 1 e leva: objetivo, contrato (seção 2), restrições de ambiente (seção 1), arquivos que o worker pode editar, critério de pronto e "saída ≠ 0 nos testes = tarefa falhou".
- Uma worktree por dev: tarefas seguintes do mesmo dev reutilizam o mesmo worker via `send`, depois de o gerente fazer o merge de `master` na worktree dele.
- O gerente não faz polling. O Orca acorda o gerente a cada `worker_done`, pergunta ou escalonamento.
- A cada marco, o gerente reporta ao usuário a tarefa concluída, a evidência e os bloqueios.
