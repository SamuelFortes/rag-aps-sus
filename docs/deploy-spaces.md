# Publicar a demo no Hugging Face Spaces

O app já está pronto para o Spaces: `app.py` na raiz é o ponto de entrada, o bloco YAML no topo do `README.md` configura o SDK (Gradio 5.16.0) e o índice versionado em `data/index/` é tudo o que a demo carrega (os PDFs não são enviados).

## Passo a passo

1. **Criar o Space.** Em https://huggingface.co/new-space, escolha um nome, SDK **Gradio**, hardware **CPU basic** (gratuito) e visibilidade pública.
2. **Cadastrar o secret e a variável.** Em *Settings → Variables and secrets*:
   - Secret `GEMINI_API_KEY` = sua chave do Google AI Studio (obrigatório).
   - Variável `GEMINI_MODEL` (opcional) = nome do modelo Gemini, para trocar o padrão de `src/config.py` sem novo commit.
3. **Enviar o repositório.** Na raiz do projeto:
   ```bash
   git remote add space https://huggingface.co/spaces/<usuario>/<nome-do-space>
   git push space master:main
   ```
   O Hugging Face pede um token de acesso com permissão de escrita como senha. Hoje o `faiss.index` tem ~7 MB, abaixo do limite de 10 MB por arquivo sem LFS; se um índice futuro passar disso e o push for recusado, rode `git lfs install` e `git lfs track "*.index"`, faça commit do `.gitattributes` e envie de novo. O `.gitignore` já impede o envio do `.env`, dos PDFs e de `data/processed/`.
4. **Acompanhar o build.** A aba *Logs* mostra a instalação do `requirements.txt` e a inicialização. No primeiro boot, o app baixa o modelo de embeddings `intfloat/multilingual-e5-base` (~1 GB), o que leva alguns minutos; depois ele fica em cache enquanto o Space estiver ativo.
5. **Testar.** Clique numa pergunta de exemplo do escopo (deve responder com citação `[documento, p. N]` e listar as fontes) e na pergunta fora do escopo (deve recusar).

## Observações

- Erros do Gemini (chave ausente, cota, indisponibilidade) aparecem para o usuário como "Não foi possível gerar a resposta agora. Tente novamente em instantes."; o erro real vai para a aba *Logs*.
- No plano gratuito, o Space hiberna após um período sem uso; a próxima visita repete o carregamento do modelo.
