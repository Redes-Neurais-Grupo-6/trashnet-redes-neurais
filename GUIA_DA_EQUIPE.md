# Guia da equipe: revisão, defesa e entrega

Este guia acompanha o projeto assistido por IA. Não comprova que os integrantes já revisaram o código, conduziram decisões ou validaram resultados. Esses passos precisam acontecer de verdade e ser registrados pela equipe. O enunciado permite IA para código, mas prevê nota zero para delegação de 100% do projeto à IA. O objetivo deste documento é ajudar a equipe a assumir o trabalho de maneira verificável, compreender os experimentos e concluir a entrega.

O PDF é obrigatório no AVA, para cada integrante, até **27 de novembro de 2026, às 23h59**. Todos devem enviar o mesmo artigo. O repositório sozinho não substitui o PDF. Não há acurácia mínima exigida: a avaliação considera execução, evidências, análise e reprodutibilidade.

## 1. O que a equipe precisa fazer pessoalmente

1. Ler o enunciado integralmente e comparar o material final com a lista de verificação da seção 6 deste guia.
2. Revisar o código, executar a avaliação dos modelos salvos e conferir se as métricas e figuras correspondem aos arquivos gerados.
3. Discutir as decisões da seção 3. Concordar com uma escolha depois de examiná-la é diferente de declarar falsamente que ela foi decidida antes da geração do projeto. Registrar o que efetivamente aconteceu.
4. Examinar as imagens de exemplo, os erros e as duplicatas conflitantes; escrever a interpretação da equipe com base no que foi observado.
5. Conferir nomes completos, matrículas e link do repositório. As matrículas `9999999` foram fornecidas para todos os integrantes e foram mantidas conforme solicitado; a equipe pode editar esses dados em `artigo/equipe.json` se necessário.
6. Publicar os artefatos, testar os links sem estar autenticado e regenerar/conferir o PDF com o endereço real do GitHub.
7. Ensaiar a defesa com o código e os dados abertos, incluindo perguntas que exponham limitações dos resultados.

Modelo de registro a ser preenchido somente após a revisão:

| Data | Integrante | Arquivo/decisão examinada | Verificação realmente realizada | Conclusão ou alteração |
|---|---|---|---|---|
| A preencher | A preencher | A preencher | A preencher | A preencher |

Não inventar commits, horas de dedicação, execuções pessoais ou concordâncias. Cada integrante deve compreender os resultados centrais, mesmo que exista divisão de tarefas.

## 2. Como conferir a execução sem refazer o treinamento

Abra o README final para instalar as dependências nas versões registradas e obter o ZIP de referência. Inicie os comandos abaixo na pasta `trashnet`; a reprodução do treinamento indica explicitamente a mudança para uma pasta nova. Não apague arquivos de resultados para forçar a execução de um comando que apontou inconsistência; examine a causa primeiro.

Para conferir os modelos e resultados entregues, sem treinar:

```bash
python -m src.prepare --root .
python -m src.evaluate --root . --evaluate
python -m src.evaluate --root . --verify-inference
python -m src.visualize --root .
```

O preparo baixa/verifica os dados e regenera os caches ausentes. `--evaluate` verifica e reutiliza os resultados já salvos; `--verify-inference` carrega os seis checkpoints, executa novamente a inferência e compara as previsões, probabilidades e métricas com os arquivos existentes, sem substituí-los. Essa conferência usa o mesmo teste e não constitui uma avaliação em dados novos. `--visualize` reconstrói as figuras. O `selection_lock.json` entregue preserva a seleção feita pela validação antes da avaliação original de teste.

Para revisar resultados já entregues:

- Execute o notebook de resultados conforme o README; ele deve reconstruir tabelas e figuras a partir dos artefatos reais.
- Confira `data/manifest.csv`: uma linha por arquivo, classes e rótulos coerentes, partições fixas.
- Confira `data/eda.json`, `data/exact_duplicates.csv` e os registros de trocas disponíveis no preparo.
- Para cada uma das seis pastas em `runs/`, examine `model.pt`, `metadata.json`, `history.csv`, as previsões de validação e as previsões/métricas de teste.
- Confirme que o hash do manifesto corresponde ao registrado nos modelos e que a seleção foi salva antes da avaliação de teste.
- Compare `reports/summary.csv` e `reports/per_class.csv` com as tabelas do artigo, incluindo suportes e unidades dos custos.
- Os notebooks recalculam métricas a partir das previsões salvas; não treinam nem executam inferência nos modelos. A nova inferência para conferência é acionada explicitamente por `--verify-inference`, conforme os comandos acima.

Para reproduzir também os seis treinamentos, mantenha a pasta entregue intacta e use uma pasta nova, fora dela, com o mesmo ambiente Python ativado:

```bash
python -m src.reproduce --destination ../trashnet_reproducao --prepare-only
cd ../trashnet_reproducao
python -m src.train --root . --experiment all
python -m src.evaluate --root . --lock
python -m src.evaluate --root . --evaluate
python -m src.visualize --root .
```

O destino deve ainda não existir. O utilitário copia o protocolo final e prepara os dados; os novos modelos e resultados são gerados nessa pasta. Os registros leves da busca original são preservados com sua proveniência, sem alegar uma nova execução da busca. Na reprodução, o treino usa treino e validação; `--lock` fixa os seis checkpoints e a seleção antes da avaliação de teste. O treino na pasta original está bloqueado: não remova esse bloqueio para contornar a proteção. O README detalha retomada, uso de ZIP local e as limitações da comparação entre ambientes.

Treinamento parcial, teste de um lote ou verificação sintética não equivalem a executar os seis experimentos completos. Confira `epochs_completed`, estado final e arquivos de cada execução antes de declarar conclusão.

## 3. Decisões metodológicas que precisam ser compreendidas

### Dados, divisão e duplicatas

O estudo usa exclusivamente as **2.527 imagens do `dataset-resized.zip`**, com classes na ordem `glass`, `paper`, `cardboard`, `plastic`, `metal`, `trash`. As proporções nominais são 70% treino, 15% validação e 15% teste, estratificadas por classe, com semente 42. Os valores inteiros efetivos estão no manifesto e na análise exploratória.

A auditoria encontrou três pares de arquivos idênticos em bytes e em pixels, porém com classes conflitantes:

| Arquivo 1 | Arquivo 2 | Problema |
|---|---|---|
| `glass/glass115.jpg` | `metal/metal91.jpg` | Mesma imagem rotulada como vidro e metal |
| `glass/glass176.jpg` | `plastic/plastic152.jpg` | Mesma imagem rotulada como vidro e plástico |
| `glass/glass389.jpg` | `plastic/plastic332.jpg` | Mesma imagem rotulada como vidro e plástico |

A política adotada preservou todas as imagens e os rótulos de origem, mantendo as contagens exigidas. Os três pares ficaram no treinamento. Após a divisão estratificada inicial, foi necessária uma troca: `plastic/plastic332.jpg` passou da validação para o treino e `plastic/plastic1.jpg`, uma imagem não duplicada da mesma classe, passou do treino para a validação. Assim, todos os totais de cada classe em cada partição permaneceram iguais. A âncora foi a primeira linha do grupo na ordem estável de classes/caminhos; a decisão não dependeu do resultado de qualquer modelo. A troca está registrada em `data/eda.json`, e os manifestos anterior e final permitem conferi-la.

Isso evita vazamento de duplicatas exatas entre partições, mas **não resolve a contradição dos rótulos**: os três pares introduzem ruído no treinamento, pois a mesma imagem está associada a classes diferentes. Nenhum desses pares ficou na validação ou no teste. Não atribua um rótulo “correto” com base na previsão do modelo nem afirme que todos os problemas do dataset foram corrigidos.

Hashes não detectam necessariamente fotografias diferentes do mesmo objeto. Como não há identificadores de instância física, a divisão não garante independência entre objetos fotografados. Essa limitação deve aparecer na discussão.

### Pré-processamento e controle

- Conversão para RGB, correção de orientação EXIF e redimensionamento diretamente da referência de 512×384; documentar o filtro usado no preparo.
- Normalização dos valores para `[0, 1]` por divisão por 255. Isso não estima estatísticas do conjunto completo.
- Ausência de aumento de dados é uma decisão válida para manter o estudo controlado; deve ser reconhecida como oportunidade de experimento futuro, sem afirmar ganho que não foi medido.
- Nas dimensões do enunciado, 128×96 significa largura × altura. Os tensores PyTorch usam canais × altura × largura.
- A MLP recebe todos os pixels RGB organizados em vetor; a CNN explora vizinhanças espaciais e compartilha filtros. Consulte `src/models.py` para explicar a arquitetura realmente executada.
- A CNN utiliza agregação espacial por média antes da classificação; por isso a quantidade de parâmetros pode permanecer igual nas três resoluções. O tamanho das ativações e o custo computacional ainda mudam.

### Pesos, checkpoint e seleção

Os pesos balanceados usam somente a contagem do treino: `w_c = N_treino / (6 × n_c_treino)`. Eles alteram a importância dos erros na perda, sem adicionar imagens nem mudar o suporte de teste. Podem elevar o recall de uma classe enquanto reduzem a precisão de outra; o resultado precisa ser medido.

Cada experimento começa com pesos novos. Os pares com/sem pesos devem usar a mesma arquitetura, semente e configuração. As três CNNs sem pesos devem preservar arquitetura, hardware e política de treinamento. Verifique hashes de inicialização e configurações nos metadados.

O projeto distingue duas decisões: o checkpoint de cada treino é recuperado pelo critério de parada configurado; depois, **os seis checkpoints salvos são comparados pelo F1 macro de validação**. Leia o campo `checkpoint_metric` para confirmar o critério exato usado na versão final. Se for menor perda de validação, o F1 usado no ranking precisa ser o calculado naquele checkpoint restaurado, não o máximo de outra época. O teste não participa dessas escolhas.

O modelo escolhido para mostrar os erros obrigatórios é o melhor por validação entre os seis. Em eventual extensão YOLO, a referência é a melhor **CNN obrigatória** por validação. São seleções diferentes.

### Busca exploratória de taxa de aprendizagem

Os registros da busca ficam separados em `search/`: `protocol.json`, `validation_trials.csv` e `selection.json`, além de configurações, históricos, metadados e previsões de validação de cada tentativa. Foram comparadas as taxas `1e-3`, `3e-4` e `1e-4`, pela média do F1 macro de validação dos checkpoints da MLP e CNN em 128×96 sem pesos. A ordem citada resolve empates exatos. O teste não é utilizado. O notebook de resultados recalcula os valores a partir das previsões de validação e confere a taxa final comum aos seis experimentos.

Essa busca foi motivada pelo comportamento inicial na validação; não deve ser descrita como um protocolo totalmente definido antes da observação dos dados de validação. Se a taxa escolhida diferiu da inicial, os seis treinos controlados foram reiniciados. Os registros preliminares ficam separados dos resultados finais. Consulte os artefatos para relatar o que efetivamente ocorreu. Não execute uma nova busca depois de examinar o teste e misture seus resultados com o estudo congelado.

### Custos e limites das conclusões

Leia os metadados para informar o hardware, número de threads, lote, número de épocas, cronômetro e definição de memória. O tempo deve excluir download/preparo e explicitar a inclusão de validação/checkpoints. Pico RSS amostrado do processo não é memória da GPU; nenhum valor ausente deve ser convertido em zero.

Uma única divisão e semente permitem a comparação pedida, mas não demonstram que a diferença se repetirá em toda execução ou em fotografias reais. Intervalos obtidos por bootstrap do teste, se presentes, caracterizam incerteza condicional aos modelos/divisão; não medem variabilidade de treinamento. Não declarar significância estatística apenas porque uma média pontual é maior.

## 4. Como escrever a análise com autoria da equipe

Partam das tabelas, curvas, matrizes e imagens realmente produzidas. Para cada afirmação, apontem o número ou painel que a sustenta. Usem pontos percentuais ao subtrair porcentagens: passar de 60% para 65% é ganhar 5 pontos percentuais.

Respondam por escrito:

1. Em 128×96, qual arquitetura teve maior F1 macro, com e sem pesos? Acurácia e F1 concordaram? Qual foi a diferença medida?
2. O que mudou em precisão, recall e F1 de `trash`? Houve perdas em alguma outra classe? Quantos exemplos sustentam a conclusão?
3. As curvas indicam aprendizagem? Há afastamento entre treino e validação? Considerem que treino usa atualizações de pesos e, quando presente, dropout; validação usa modo de avaliação.
4. Qual resolução ofereceu melhor relação entre desempenho e tempo? O tempo total foi afetado pelo número de épocas até a parada?
5. Quais confusões mais ocorreram? Quais imagens concretas ajudam a explicar o padrão? Evitem explicações visuais que ninguém conferiu.
6. O que limita uso em cenas reais, com múltiplos objetos, iluminação diferente, fundo complexo e resíduos ausentes no dataset?
7. Como os rótulos contraditórios afetam interpretação das métricas? Que auditoria adicional seria útil?

Para o estudo histórico, não misturem protocolos: Yang e Thung relatam SVM de 63% com seis classes e split 70/30; a CNN do relatório atinge 22%, exclui `trash`, usa cinco classes, split 70/13/17 e entrada 256×256. O README informa aproximadamente 75% em continuação posterior. Não apresentar esses valores como resultados da equipe ou como comparação controlada com o presente estudo. Fontes: [relatório](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf) e [repositório oficial](https://github.com/garythung/trashnet).

Uma declaração de uso de IA, se incluída, deve descrever somente o que ocorreu. Exemplo de estrutura a preencher: “Foi utilizada assistência de IA em [tarefas reais]. Os integrantes realizaram [ações efetivamente realizadas], verificaram [evidências reais] e assumem [decisões realmente discutidas].” Não manter campos genéricos nem assinar revisão inexistente.

## 5. Perguntas para a defesa

| Pergunta | O que a resposta precisa demonstrar |
|---|---|
| Por que acurácia sozinha é insuficiente? | A contribuição maior das classes frequentes e a possibilidade de fracasso em `trash` ficar escondida. |
| Como calcular o F1 macro? | Calcular F1 por classe e fazer a média aritmética dos seis valores; não ponderar pelo suporte. |
| O que é suporte? | Número de exemplos reais da classe no conjunto avaliado. |
| O que significa `zero_division=0`? | Uma convenção para métricas indefinidas; não prova desempenho adequado. |
| Por que preservar `trash`? | Faz parte dos seis experimentos e permite estudar a classe minoritária. |
| Como os pesos foram calculados? | Fórmula, contagens apenas do treino e localização dos valores no registro da execução. |
| Pesos de classe duplicam imagens? | Não; alteram a perda. Reamostragem seria outra intervenção. |
| Como se evitou vazamento? | Manifesto compartilhado, treino/validação/teste separados, duplicatas exatas agrupadas e teste fora da escolha de configuração. |
| Por que manter os rótulos conflitantes? | Preservação do conjunto requerido, tratamento transparente do risco e ausência de anotação externa confiável para corrigi-los. |
| Como funciona a troca de duplicatas? | Mesma partição para o grupo; troca com singleton da mesma classe; contagens classe×partição preservadas. |
| Por que usar uma MLP como comparação? | Estabelecer referência com pixels vetorizados e observar o papel das operações espaciais da CNN. |
| Por que não aplicar softmax antes de CrossEntropyLoss? | A função recebe logits; probabilidades são obtidas separadamente para apresentação/inferência. |
| Como a CNN mantém o número de parâmetros? | Filtros e classificador fixos com pooling global; resolução muda dimensões espaciais, não os pesos aprendidos. |
| Qual modelo foi escolhido e por quê? | Mostrar o F1 macro de validação do checkpoint salvo; não justificar a escolha pelo teste. |
| O que determinou a parada? | Mostrar a configuração, curva de validação, época selecionada e épocas executadas. |
| O ganho de resolução compensou o custo? | Citar dados reais de F1/acurácia, tempo médio, total, épocas e memória; reconhecer ajustes de lote se ocorreram. |
| O modelo está pronto para uso real? | Distinguir avaliação neste dataset de validação externa em outras condições. |
| Como reproduzir as métricas? | Abrir notebook, manifesto, checkpoint e previsões; explicar o procedimento sem novo treinamento. |
| Por que 22%, 63% e 75% não são comparação direta? | Identificar fontes, classes, splits, resolução e continuação posterior do repositório. |
| Que parte vocês realmente revisaram? | Responder com registros e compreensão própria, sem atribuir à equipe trabalho que não realizou. |

## 6. Checklist de entrega e cobertura da rubrica

Deixado em branco intencionalmente: os integrantes devem marcar apenas após conferir o material final.

| Critério | Evidências a conferir | Valor |
|---|---|---:|
| [ ] Dados | Integridade, dimensões, canais, exemplos, contagens, tratamento das duplicatas, RGB/normalização, manifesto estratificado e ausência de vazamento exato | 0,75 |
| [ ] Modelos | MLP e CNN treinadas, descritas e justificadas; seis checkpoints reais | 1,00 |
| [ ] Experimentos | Dois pares com/sem pesos em 128×96; CNN sem pesos em 128×96, 256×192 e 512×384; controle e custos comparáveis | 1,00 |
| [ ] Resultados | Acurácia/F1 macro; precisão/recall/F1/suporte por classe; seis matrizes; curvas e gráficos exigidos; painel de erros do selecionado | 1,00 |
| [ ] Artigo | Estrutura, clareza, interpretação de desempenho/custo/limitações e comparação histórica contextualizada | 0,75 |
| [ ] Reprodutibilidade | README, versões, código, notebooks, manifesto, configurações, históricos, métricas, previsões e seis modelos acessíveis | 0,50 |

Figuras/tabelas que precisam estar **no artigo e no notebook**:

- [ ] Barras com contagens por classe em treino, validação e teste.
- [ ] Pelo menos duas imagens de cada classe.
- [ ] Uma imagem apresentada nas três resoluções.
- [ ] Perda e acurácia de treino/validação por época para todos os seis experimentos.
- [ ] Seis matrizes com contagens e porcentagens por linha, classes reais nas linhas e previstas nas colunas, na ordem exigida.
- [ ] Desempenho da CNN sem pesos por resolução: acurácia e F1 macro.
- [ ] Tempo médio por época e tempo total da CNN sem pesos por resolução.
- [ ] Tabela comparativa de métricas e custos dos seis modelos.
- [ ] Precisão, recall, F1 e suporte para as 36 combinações de modelo e classe.
- [ ] Até seis erros do modelo escolhido por validação, com classe real, prevista e probabilidade; menos que seis quando não houver mais erros.

Conferência final do PDF:

- [ ] Até dez páginas no total, incluindo figuras, tabelas e referências; uma coluna; sem capa separada.
- [ ] Primeira página com título, nomes completos, matrículas conferidas, resumo e GitHub real.
- [ ] Introdução, materiais e métodos, resultados, discussão, conclusão e referências.
- [ ] Figuras/tabelas numeradas, fontes indicadas e interpretação no texto.
- [ ] Valores, unidades, arredondamentos e nomes dos modelos coerentes com os arquivos.
- [ ] Fontes externas citadas e referenciadas, incluindo dataset, artigo e bibliotecas utilizadas.
- [ ] Uso de IA descrito sem declarações falsas sobre revisão ou autoria.
- [ ] Arquivo aberto em outro visualizador para detectar texto cortado ou gráficos ilegíveis.
- [ ] Mesmo PDF enviado por todos os integrantes no AVA.

O extra YOLO é opcional e só deve ser apresentado como executado se houver treino real, modelo acessível, métricas completas, matriz, curvas, custos e discussão. Não substitui nenhum experimento obrigatório. A nota máxima continua sendo 5,0.

## 7. Publicação no GitHub depois da revisão final

1. Criem um repositório da equipe com nome claro. Confiram a visibilidade/acesso exigido para o professor. Não é necessário incluir o cache de imagens nem o ambiente Python.
2. Revisem o `.gitignore`. Excluam somente os arrays regeneráveis `data/cache/*/images.npy`, preservando os três arquivos `data/cache/*/metadata.json`, cujos hashes integram o bloqueio de seleção. Não excluam a pasta `data/cache/` inteira. O ZIP do dataset, ambiente virtual, `__pycache__` e checkpoints temporários de retomada também não precisam acompanhar o código publicado. O download/preparo deve continuar documentado. **Os seis modelos finais precisam estar acessíveis**, no repositório ou por links livres descritos no README.
3. Incluam código, configuração, dependências, README, este guia, notebooks, figuras, tabelas, manifesto, registros e modelos finais. Preservem citações e informações de origem do TrashNet.
4. Se optarem pelo Git local, iniciem o repositório na pasta do projeto, confiram os arquivos preparados para commit e conectem a URL real do repositório. Não executem `git add .` antes de conferir as exclusões; caches podem ser grandes.
5. Arquivos acima do limite do GitHub precisam de outro mecanismo. O enunciado permite Google Drive público com download livre e instruções. A documentação do GitHub informa bloqueio de arquivos acima de 100 MiB em Git comum e limite de 25 MiB para upload pelo navegador. [Limites oficiais do GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
6. Para cada modelo externo, incluam nome, tamanho, SHA-256, link e pasta onde salvar. Testem o link em janela sem login; “solicitar acesso” não é download livre.
7. Atualizem o link no artigo e gerem o PDF final novamente. Não basta editar o README.
8. Em uma pasta nova, obtenham uma cópia do repositório, sigam o README e reproduzam a avaliação/notebook. Confiram que não dependem de caminhos particulares do computador de quem criou o projeto.
9. Registrem um commit ou versão final correspondente ao PDF entregue. Preservem os artefatos usados na avaliação; mudanças posteriores devem ficar claramente separadas.

## 8. Fontes e normas de apresentação

Use as referências já conferidas no artigo, apenas quando citadas no texto. As versões identificadas na pesquisa foram NBR 6022:2018 para artigo, NBR 10520:2023 para citações e NBR 6023:2025 para referências. A indicação das versões foi verificada na [Biblioteca da UFES](https://biblioteca.ufes.br/manuais-de-normalizacao). O professor exige uma coluna, ausência de capa e até dez páginas; essas regras específicas prevalecem na atividade.

Revisem citações autor-data, autoria das imagens, URL/data de acesso e correspondência entre citações e referências. Não atribuam DOI, volume ou páginas inexistentes ao relatório de curso de Yang e Thung. A data de acesso deve refletir a consulta realizada, não uma data futura presumida.
