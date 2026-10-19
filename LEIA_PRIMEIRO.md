# Comece por aqui

O pacote reúne seis treinamentos reais no TrashNet, seus modelos salvos, históricos, previsões, tabelas e figuras. A seleção foi congelada pela validação antes do teste. Os dois notebooks foram executados por completo; a inferência dos seis modelos foi repetida e conferida com os resultados salvos.

## Para abrir e revisar

1. Leia `artigo/artigo_redes_neurais.pdf`: são dez páginas, incluindo todas as figuras, tabelas e referências.
2. Abra `notebooks/01_analise_exploratoria.ipynb` e `notebooks/02_resultados.ipynb`. As saídas já estão salvas.
3. Use `README.md` para instalar as dependências e reproduzir as verificações sem retreinar. As imagens são baixadas pelo preparo; os seis modelos finais já estão em `runs/`.
4. Consulte `GUIA_DA_EQUIPE.md` para revisar a metodologia, discutir resultados e preparar a defesa. A auditoria automatizada está em `reports/final_audit.json`.

## Antes da submissão

- Substituam as matrículas provisórias `9999999` em `artigo/equipe.json`.
- Criem o repositório GitHub, publiquem os arquivos deste pacote e preencham `github_url` no mesmo JSON. O enunciado exige GitHub; Colab pode executar código, mas não substitui o repositório.
- Realizem a revisão e a discussão de fato. O enunciado permite IA para código e exige participação da equipe; acrescentar nomes não substitui essa participação.
- Revisem os textos em `artigo/conteudo.json`. Só marquem `revisao_da_equipe_concluida` como `true` depois de concluir a revisão.
- Na raiz do projeto, executem `python -m src.build_article --root .`. Confiram o novo PDF, inclusive o link, e enviem o mesmo documento no AVA por todos os integrantes.

Prazo indicado no enunciado: **27/11/2026, às 23h59**.

## O que publicar

Publiquem a pasta `trashnet` extraída do ZIP, incluindo os seis `runs/*/model.pt`, os notebooks, a documentação, o artigo, os registros e os metadados dos caches. O ZIP já omite imagens regeneráveis, caches volumosos e checkpoints temporários. Não publiquem somente o ZIP: o professor precisa navegar pelos códigos e notebooks no repositório.

Não alterem modelos, previsões ou registros congelados para editar o artigo. O README explica como iniciar uma reprodução separada quando desejarem treinar novamente.
