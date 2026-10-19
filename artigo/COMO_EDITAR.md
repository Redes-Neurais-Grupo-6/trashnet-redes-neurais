# Ajustar o artigo antes da entrega

1. Abra `equipe.json` e substitua as matrículas `9999999` pelas corretas.
2. Após publicar o repositório, preencha `github_url` com a URL pública real.
3. A equipe deve conferir os resultados, discutir as decisões e revisar o texto. Só então altere `revisao_da_equipe_concluida` para `true`.
4. Para revisar a redação, edite os textos de `conteudo.json`. As tabelas contêm os resultados medidos; qualquer alteração numérica deve ser conferida nos CSVs de `reports/` e nos registros de `runs/`.
5. Na raiz do projeto, com as dependências instaladas, execute:

```bash
python -m src.build_article --root .
```

O gerador grava `artigo/artigo_redes_neurais.pdf`, uma cópia de leitura em `artigo/artigo_editavel.md` e o diagnóstico de espaço por página em `artigo/layout_check.json`. O Markdown é uma exportação para facilitar a revisão; a fonte utilizada pelo gerador é o JSON.

O gerador recusa texto que ultrapasse a página. Depois de qualquer edição, abra o PDF e confira as dez páginas, as figuras, as matrículas e o link. Não basta marcar o campo de revisão sem realizar a revisão.

Os arquivos de identificação e redação são separados dos modelos e dos resultados congelados: corrigir nomes, matrículas e redação não altera os treinamentos. Se uma interpretação for revisada, confira se ela continua sustentada pelos resultados registrados.
