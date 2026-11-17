# Publicar a demo no Hugging Face Spaces

Demo publicada: https://huggingface.co/spaces/Samuka21/assistente-aps-sus

O app já está pronto para o Spaces:
- o `app.py` na raiz é o ponto de entrada;
- o bloco YAML no topo do `README.md` configura o SDK (Gradio 5.16.0);
- o índice versionado em `data/index/` é tudo o que a demo carrega. Os PDFs não são enviados.

## Hardware: ZeroGPU

Contas gratuitas não podem mais hospedar Spaces Gradio em CPU básico; isso exige assinatura PRO, e a API responde `402 Payment Required`. Por isso o Space roda em **ZeroGPU**, que é gratuito.

No ZeroGPU, o Space precisa de pelo menos uma função com `@spaces.GPU`. Em `src/retrieval.py`, `encode_query` (o embedding da pergunta) recebe esse decorador quando o pacote `spaces` existe. A GPU é alocada só durante essa chamada, e o resto da busca e a chamada ao Gemini rodam em CPU. Fora dos Spaces, o pacote `spaces` não está instalado e o app roda normalmente em CPU.

## Passo a passo

1. **Criar o Space.** Em https://huggingface.co/new-space, escolha um nome, SDK **Gradio**, template **Blank**, licença MIT e visibilidade pública.
2. **Cadastrar o secret.** Em *Settings → Variables and secrets*:
   - Secret `GEMINI_API_KEY` = sua chave do Google AI Studio (obrigatório).
   - Variável `GEMINI_MODEL` (opcional) = outro modelo Gemini, sem precisar de novo commit.
3. **Fazer login.** Crie um token *Write* em https://huggingface.co/settings/tokens e rode `.venv\Scripts\hf.exe auth login`.
4. **Enviar só os arquivos da demo.** Use `upload_folder`, não `git push` nem `hf upload`. O `hf upload` tenta criar o repositório e recebe o 402.
   ```powershell
   .venv\Scripts\python.exe -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='<usuario>/<space>', repo_type='space', folder_path='.', allow_patterns=['app.py','README.md','requirements.txt','src/*.py','data/index/*.jsonl','data/index/*.index','docs/demo.gif'], commit_message='Atualiza demo')"
   ```
   O `upload_folder` envia os binários (`faiss.index`, GIF) pelo armazenamento de arquivos grandes automaticamente.
5. **Acompanhar o build.** A aba *Logs* mostra a instalação do `requirements.txt` e a inicialização. O primeiro boot baixa o modelo de embeddings `intfloat/multilingual-e5-base` (~1 GB) e leva alguns minutos.
6. **Testar.** Uma pergunta do escopo deve responder com citação `[documento, p. N]` e listar as fontes. A pergunta fora do escopo deve ser recusada.

## Observações

- Erros do Gemini (chave ausente, cota, indisponibilidade) aparecem para o usuário como "Não foi possível gerar a resposta agora. Tente novamente em instantes.". O erro real vai para a aba *Logs*.
- O ZeroGPU tem cota diária de GPU por visitante. Cada pergunta usa poucos segundos, e a primeira de cada sessão espera a alocação da GPU.
- No plano gratuito, o Space hiberna depois de um período sem uso, e a próxima visita repete o carregamento do modelo.
