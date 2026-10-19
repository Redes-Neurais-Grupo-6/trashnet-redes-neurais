# Classificação de resíduos com redes neurais

Estudo experimental do TrashNet que compara MLP e CNN, pesos de classe e resolução de entrada. O projeto reúne os seis modelos obrigatórios, divisão reproduzível, registros de treinamento, avaliação, figuras, notebooks e artigo. Os resultados numéricos têm como fonte os arquivos de `runs/` e `reports/`, evitando tabelas manuais divergentes.

**GitHub e Colab:** o enunciado exige um repositório GitHub com link na primeira página do artigo. Colab é uma opção para executar código, mas não substitui esse repositório. Os notebooks dependem da pasta completa do projeto (incluindo `src/`, `runs/` e dados preparados); abrir apenas um `.ipynb` isolado não reproduz o estudo.

**Antes da entrega:** conferir as matrículas provisórias `9999999`, preencher o endereço real do GitHub em `artigo/equipe.json`, revisar o trabalho com os integrantes e gerar novamente o PDF. A publicação do repositório e a revisão humana não são comprovadas por estes arquivos. Leia o [Guia da equipe](GUIA_DA_EQUIPE.md), que inclui perguntas para defesa, registro de revisão e roteiro de publicação.

## 1. Começar pelos resultados salvos

Após instalar as dependências e preparar os dados, a conferência dos modelos entregues não exige treinamento:

```bash
python -m src.prepare --root .
python -m src.evaluate --root . --evaluate
python -m src.evaluate --root . --verify-inference
python -m src.visualize --root .
```

Execute os comandos na pasta que contém este README. `--evaluate` verifica e reutiliza os resultados existentes; `--verify-inference` carrega novamente os seis modelos, executa inferência e compara rótulos, probabilidades e métricas com os arquivos salvos, sem alterá-los. O segundo comando depende do `selection_lock.json` entregue. O primeiro acesso baixa o dataset e produz os caches, caso ainda não estejam presentes.

Consulte:

- [Artigo em PDF](artigo/artigo_redes_neurais.pdf).
- [Notebook de análise exploratória](notebooks/01_analise_exploratoria.ipynb).
- [Notebook de resultados](notebooks/02_resultados.ipynb).
- [Métricas e custos dos seis experimentos](reports/summary.csv).
- [Precisão, recall, F1 e suporte por classe](reports/per_class.csv).
- [Configuração final](configs/experiments.json) e [seleção congelada](selection_lock.json).

## Resultados medidos

| Experimento | Acurácia de teste | F1 macro de teste | Tempo total |
|---|---:|---:|---:|
| MLP 128×96 sem pesos | 44,47% | 0,3583 | 56,7 s |
| MLP 128×96 com pesos | 38,42% | 0,3304 | 60,4 s |
| CNN 128×96 sem pesos | 61,84% | 0,5241 | 49,2 s |
| CNN 128×96 com pesos | 58,16% | 0,5718 | 50,9 s |
| CNN 256×192 sem pesos | 59,21% | 0,5131 | 216,5 s |
| CNN 512×384 sem pesos | 62,11% | 0,5223 | 1258,6 s |

A **CNN 128×96 com pesos** foi selecionada pelo maior F1 macro de validação (0,5591), antes de consultar o teste. Seu F1 macro de teste foi 0,5718 (IC95% bootstrap: 0,5243–0,6181). Os pesos melhoraram o recall de `trash` para 70%, mas sua precisão ficou em 28,57%; a discussão inclui os erros nas demais classes.

A CNN 512×384 sem pesos teve apenas um acerto adicional frente à CNN 128×96 sem pesos (236 contra 235 em 380), com F1 macro ligeiramente menor e 25,59 vezes o tempo total. Neste protocolo, a maior resolução não compensou o custo. Não houve novo ajuste depois de consultar o teste.

## 2. Ambiente e instalação

Ambiente utilizado: **Python 3.12.14, Linux x86-64, CPU Intel Core i3-1125G4, 4 threads PyTorch, sem GPU**. O script usa tensores `float32`. As versões diretas estão fixadas em [requirements.txt](requirements.txt); [environment.json](environment.json) registra o ambiente da execução.

Com Python 3.12 instalado, no Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
python -m pip install -r requirements.txt
python -m pip check
```

Instale primeiro o pacote CPU do PyTorch pelo índice indicado. O sufixo `+cpu` faz parte da versão registrada. Os comandos foram concebidos para este ambiente; outra plataforma pode exigir uma distribuição PyTorch compatível e deve registrar essa diferença.

| Dependência | Versão |
|---|---:|
| PyTorch | 2.14.1+cpu |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| Matplotlib | 3.11.2 |
| psutil | 7.2.2 |
| nbformat / nbclient | 5.11.1 / 0.11.0 |
| ipykernel | 7.4.0 |
| ReportLab / pypdf | 4.4.9 / 6.10.0 |

Reserve alguns gigabytes de disco para o ambiente, os dados e os caches. Os três caches de imagens somam aproximadamente 1,96 GB, além do ZIP e das imagens extraídas. O cache é mapeado em memória; o consumo RSS inclui páginas acessadas desse mapeamento. O treinamento dos experimentos é sequencial para preservar as medições de custo.

## 3. Dados, integridade e divisão

Use exclusivamente o [`dataset-resized.zip` oficial](https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip?download=true), disponibilizado pelo [TrashNet](https://github.com/garythung/trashnet). Ele contém 2.527 imagens RGB de 512×384 pixels. Não combine essa versão com os originais do mesmo repositório.

Download e preparação automáticos:

```bash
python -m src.prepare --root .
```

Para usar um ZIP já obtido, forneça seu caminho real:

```bash
python -m src.prepare --root . --zip /caminho/dataset-resized.zip
```

SHA-256 esperado do ZIP:

```text
c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c
```

SHA-256 do manifesto final de referência:

```text
988a0a94bf617b8f3ab88e9d6da355001cb8fe2dc59f98004779bc245b9adf7f
```

O preparo verifica o ZIP, a integridade das imagens, contagens, dimensões e modos de cor. Corrige orientação EXIF, converte para RGB e gera cada resolução diretamente da referência com Pillow/LANCZOS. Os tensores têm ordem canais×altura×largura; a normalização é `float32 / 255`. Não há aumento de dados, reamostragem nem normalização estimada no teste.

A divisão nominal é 70/15/15, estratificada, com semente 42. Os valores inteiros efetivos são:

| Classe | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| glass | 350 | 75 | 76 | 501 |
| paper | 416 | 89 | 89 | 594 |
| cardboard | 282 | 61 | 60 | 403 |
| plastic | 337 | 72 | 73 | 482 |
| metal | 287 | 61 | 62 | 410 |
| trash | 96 | 21 | 20 | 137 |
| **Total** | **1.768** | **379** | **380** | **2.527** |

Foram encontrados três pares de imagens idênticas com rótulos conflitantes. Os grupos foram mantidos na mesma partição, preservando os rótulos originais e todas as contagens classe×partição; uma troca determinística foi necessária. Os três pares ficaram no treino. A política impede vazamento de duplicatas exatas, mas não resolve a contradição de rótulos nem identifica todas as fotos diferentes de um mesmo objeto. Consulte `data/exact_duplicates.csv`, `data/eda.json` e `data/manifest_before_grouping.csv`.

`data/manifest.csv` é compartilhado pelos seis experimentos. O preparo recusa sobrescrever um manifesto existente diferente. Cada cache declara o hash do manifesto, seu formato e seu próprio hash em `data/cache/LxA/metadata.json`. A avaliação verifica a integridade do conteúdo dos caches antes da inferência.

## 4. Experimentos e decisões de treinamento

As dimensões da tabela são **largura × altura**:

| Identificador | Rede | Entrada | Pesos de classe |
|---|---|---|---|
| `mlp_128_unweighted` | MLP | 128×96 | Não |
| `mlp_128_weighted` | MLP | 128×96 | Sim |
| `cnn_128_unweighted` | CNN | 128×96 | Não |
| `cnn_128_weighted` | CNN | 128×96 | Sim |
| `cnn_256_unweighted` | CNN | 256×192 | Não |
| `cnn_512_unweighted` | CNN | 512×384 | Não |

A MLP usa vetor RGB → 128 → 64 → 6, ReLU e dropout 0,2, com **4.727.366 parâmetros**. A CNN usa três blocos Conv3×3/BatchNorm/ReLU/MaxPool2, filtros 16/32/64, stride 2 somente na primeira convolução, média espacial global e classificador de seis saídas. São **24.086 parâmetros** em qualquer uma das três resoluções. Os modelos produzem logits; softmax é aplicado somente para obter probabilidades na avaliação.

Todos usam Adam, lote 32, no máximo 25 épocas e parada após cinco épocas sem redução estrita da perda de validação. O checkpoint `model.pt` corresponde à menor perda de validação. A taxa de aprendizagem final está em `configs/experiments.json` e nas configurações de cada execução. Nos experimentos ponderados, `w_c = N_treino / (6 × n_c_treino)`; nenhuma contagem de validação/teste entra nos pesos.

Uma busca exploratória limitada comparou as taxas 0,001, 0,0003 e 0,0001 em MLP128 e CNN128 sem pesos. A escolha usa a média dos dois F1 macro de validação dos checkpoints restaurados; o primeiro candidato vence empate exato. O diagnóstico inicial e a busca antecedem o teste. Protocolo, tentativas e decisão estão em `search/protocol.json`, `search/validation_trials.csv` e `search/selection.json`. Essa busca é separada dos seis experimentos controlados finais. Ela não identifica a melhor taxa possível para qualquer orçamento ou arquitetura.

Os experimentos reiniciam os geradores e os pesos com semente 42; pares comparáveis preservam os mesmos hashes de inicialização e a ordem dos lotes. Os metadados permitem verificar esse controle. O F1 macro usado para escolher o modelo final é calculado **no checkpoint restaurado**, não no pico de F1 de outra época.

O cronômetro `perf_counter` inclui treino, validação de cada época e gravação dos checkpoints; exclui download, preparação, inicialização e avaliação final do checkpoint restaurado. O pico RSS absoluto do processo é amostrado a cada 50 ms por `psutil`. O tamanho publicado refere-se a `model.pt`, sem estado do otimizador. As perdas ponderadas e não ponderadas têm objetivos diferentes; sua magnitude isolada não é critério de comparação de desempenho.

## 5. Reproduzir o treinamento em uma pasta nova

O projeto entregue já possui seleção e teste registrados. O treino nessa pasta é bloqueado para preservar o protocolo. **Não apague o bloqueio nem substitua os resultados originais.** Para reproduzir a configuração final, crie uma pasta nova pelo utilitário:

```bash
python -m src.reproduce --destination ../trashnet_reproducao --prepare-only
```

Se o ZIP já estiver disponível:

```bash
python -m src.reproduce --destination ../trashnet_reproducao --zip /caminho/dataset-resized.zip --prepare-only
```

O destino precisa não existir e deve ficar fora da pasta original. São copiados código, configurações finais, testes, dependências e manifesto; são preparados os dados. Modelos, previsões de teste, resultados finais e bloqueio original não são copiados como resultados novos. A origem fica documentada em `provenance/`. Os registros leves da busca original são preservados em `search/`, identificados por `search/ORIGIN.md`, para a auditoria do notebook; eles não representam uma nova execução da busca.

Depois, mantendo o mesmo ambiente Python ativado:

```bash
cd ../trashnet_reproducao
python -m src.train --root . --experiment all
python -m src.evaluate --root . --lock
python -m src.evaluate --root . --evaluate
python -m src.visualize --root .
```

Para executar essas etapas automaticamente após o preparo, omita `--prepare-only` no comando do utilitário. A reprodução utiliza a taxa final já escolhida; não repete a busca por padrão. Os seis treinos são executados em processos separados e sequenciais.

Em uma reprodução interrompida **antes do bloqueio**, retome o treinamento com:

```bash
python -m src.train --root . --experiment all --resume
```

`last.pt` preserva otimizador, geradores aleatórios, ordem de lotes e controle da parada. Ele é gerado durante o treino, mas não precisa integrar o pacote de entrega. `model.pt` é o checkpoint escolhido para inferência. A retomada pode subcontabilizar o custo da última gravação antes da interrupção, limitação registrada nos metadados; execuções contínuas não têm essa limitação.

Essa reprodução confirma o protocolo já avaliado; não constitui um novo teste independente. Alterações motivadas pelos resultados do teste devem ser declaradas como outro estudo exploratório, com validação externa nova para sustentar conclusões de generalização. Determinismo é solicitado no PyTorch, mas igualdade numérica entre versões, plataformas ou dispositivos diferentes não é garantida; registre o ambiente e compare as evidências. Veja a [documentação de reprodutibilidade](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## 6. Avaliação, figuras e notebooks

`--lock` registra a seleção por maior F1 macro de validação entre os seis modelos, além de hashes de pesos, configuração, históricos, manifesto e código central. Empates exatos seguem a ordem fixa dos experimentos. Depois de criado, o bloqueio impede novos treinos e mudanças nos arquivos acompanhados.

`--evaluate` calcula acurácia, F1 macro, precisão/recall/F1/suporte por classe, matrizes com contagens e porcentagens por linha e intervalos por bootstrap estratificado. São 2.000 reamostragens, semente 20261127. Esses intervalos medem incerteza do conjunto de teste condicionada aos modelos; não medem variabilidade entre sementes de treinamento.

As seis classes aparecem sempre nas métricas, na ordem `glass`, `paper`, `cardboard`, `plastic`, `metal`, `trash`. Métricas indefinidas recebem zero por `zero_division=0`. O painel de erros usa até seis previsões incorretas do modelo selecionado pela validação, ordenadas por maior probabilidade prevista; empates seguem o caminho do arquivo. A probabilidade softmax não é garantia de confiança calibrada.

As figuras são geradas em PNG a 300 dpi e SVG:

```bash
python -m src.visualize --root .
```

Para produzir somente distribuição, exemplos e resoluções antes do treinamento:

```bash
python -m src.visualize --root . --eda-only
```

Os notebooks podem ser abertos em um editor compatível com Jupyter. Para executá-los pelo terminal usando as dependências incluídas, registre um kernel no ambiente virtual e use `nbclient`:

```bash
python -m ipykernel install --sys-prefix --name trashnet --display-name "Python (TrashNet)"
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path.cwd().resolve()
for filename in ("01_analise_exploratoria.ipynb", "02_resultados.ipynb"):
    path = root / "notebooks" / filename
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=1200, kernel_name="trashnet",
                   resources={"metadata": {"path": str(root)}}).execute()
    nbformat.write(notebook, path)
    print("Executado:", path.name)
PY
```

Prepare os dados antes de executar os notebooks; o de resultados também requer os seis resultados congelados. Eles apresentam os dados e registros existentes, sem retreinar os modelos. Para comprovar uma nova inferência dos pesos salvos, execute explicitamente `--verify-inference`.

Auditoria completa do pacote (após gerar o artigo e executar os notebooks):

```bash
python -m src.audit --root .
```

Os registros da conferência de inferência e dos testes estão em `reports/verification_log.txt` e `reports/protocol_tests_log.txt`.

Testes do protocolo:

```bash
python -m unittest discover -s tests -v
```

Os testes usam dados sintéticos isolados para verificar invariantes do programa, incluindo inicialização, pesos calculados somente no treino, consistência do cache, bloqueio e retomada. Eles complementam a avaliação real do TrashNet.

## 7. Arquivos e modelos

```text
artigo/                 PDF e equipe.json para identificação
configs/                configuração dos seis experimentos finais
data/                   manifesto, inspeção e registro de duplicatas
  cache/LxA/            metadata.json e images.npy regenerável
  raw/dataset-resized/  imagens extraídas do ZIP oficial
notebooks/              análise exploratória e resultados
reports/                tabelas agregadas e figuras PNG/SVG
runs/ID/                modelo, configuração, histórico e previsões
search/                 registros da busca exploratória por validação
src/                    preparação, redes, treino, avaliação e figuras
tests/                  verificações automatizadas do protocolo
selection_lock.json     decisões congeladas antes do teste
requirements.txt        dependências diretas fixadas
environment.json        versões usadas na execução
GUIA_DA_EQUIPE.md        revisão, defesa e publicação
```

Cada diretório final de `runs/` contém:

| Arquivo | Conteúdo |
|---|---|
| `model.pt` | `state_dict` PyTorch do checkpoint escolhido |
| `config.json` | Hiperparâmetros efetivos |
| `metadata.json` | Épocas, custos, hardware, hashes e validação restaurada |
| `history.csv` | Perda/acurácia de treino/validação e tempo por época |
| `val_predictions.csv` | Previsões usadas para seleção por validação |
| `predictions.csv` | Previsões e probabilidades no teste |
| `metrics.json` | Métricas completas, matriz, suportes e intervalos |

Cada MLP ocupa aproximadamente **18,9 MB** em pesos; cada CNN, aproximadamente **0,1 MB**. Os tamanhos e hashes exatos estão nos metadados e no bloqueio. Os seis `model.pt` acompanham o projeto; não é necessário treinar para obtê-los. Eles são carregados com `torch.load(..., weights_only=True)`. Não trate `last.pt`, que também contém estado de treinamento, como o modelo final.

O pacote de distribuição exclui imagens originais, ZIP do dataset, caches `images.npy`, ambiente virtual, `__pycache__` e checkpoints `last.pt`. Inclui os metadados dos caches necessários para verificar o estudo. `prepare` regenera os dados ausentes a partir da referência.

## 8. Artigo, identificação e publicação

Os nomes, matrículas, título e endereço do repositório são lidos de `artigo/equipe.json`. As matrículas `9999999` são provisórias e `github_url` precisa receber a URL real após publicação. O campo de revisão só deve ser marcado quando a equipe realmente tiver revisado e compreendido o trabalho.

Depois de preencher a identificação e conferir os resultados:

```bash
python -m src.build_article --root .
```

O artigo é salvo em `artigo/artigo_redes_neurais.pdf`. Abra o PDF gerado, confira as páginas, legibilidade e identificação, e envie o mesmo arquivo por todos os integrantes no AVA. O prazo do enunciado é **27 de novembro de 2026, às 23h59**. A entrega do PDF é obrigatória; um repositório disponível não a substitui.

O [Guia da equipe, seção 7](GUIA_DA_EQUIPE.md#7-publicação-no-github-depois-da-revisão-final) descreve a publicação. Inclua os seis modelos, código, notebooks, tabelas, figuras, manifesto e registros; preserve as exclusões dos dados regeneráveis. Teste o acesso sem autenticação e use no artigo a URL realmente publicada. Não há endereço fictício de GitHub neste projeto.

## 9. Limitações e uso de IA

O dataset apresenta objetos principalmente isolados e fundos controlados; não demonstra desempenho em cenas reais com múltiplos resíduos. A classe `trash` tem apenas 20 exemplos no teste, tornando seu recall sensível a poucas imagens. Há rótulos contraditórios em duplicatas do treino, possível correlação entre fotografias de um mesmo objeto e apenas uma divisão/semente. O estudo não executa o bônus opcional YOLO.

Este material foi produzido com assistência de IA. Não se afirma que os integrantes já tenham executado, revisado ou decidido pessoalmente todas as etapas. O enunciado permite assistência para código, mas exige condução, compreensão e interpretação pela equipe e prevê nota zero para delegação integral. A revisão real, a discussão das escolhas e a conferência das evidências são parte da conclusão do trabalho. Não basta acrescentar os nomes a um material que ninguém compreendeu.

## 10. Fontes

- THUNG, Gary; YANG, Mindy. [TrashNet: repositório e descrição dos dados](https://github.com/garythung/trashnet).
- YANG, Mindy; THUNG, Gary. [Classification of trash for recyclability status](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf). Relatório CS229, Stanford, 2016. O relatório e os resultados posteriores do README têm protocolos diferentes; não são comparação controlada com este estudo.
- PYTORCH CONTRIBUTORS. [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), [Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html) e [Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
- SCIKIT-LEARN DEVELOPERS. [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html), [compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) e [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
- UNIVERSIDADE FEDERAL DO ESPÍRITO SANTO. [Manuais de normalização](https://biblioteca.ufes.br/manuais-de-normalizacao), consultados para orientar as versões das normas de apresentação e referências.

Fontes consultadas em 16 e 17 de outubro de 2026. As imagens de exemplos e erros provêm do TrashNet; gráficos e tabelas são derivados das execuções registradas. A autoria e as condições do conjunto de dados permanecem atribuídas aos autores originais.
