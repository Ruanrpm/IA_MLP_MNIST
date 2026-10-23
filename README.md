# Reconhecimento de dígitos manuscritos com MLP

Projeto de Inteligência Artificial em Python. A rede totalmente conectada, o forward propagation, a entropia cruzada e o Backpropagation são implementados manualmente com NumPy. Matplotlib gera as figuras. Não há TensorFlow, Keras, PyTorch, Scikit-learn, modelos pré-treinados ou diferenciação automática.

## Instalação

### Executar com Docker (recomendado para levar à faculdade)

Requer Docker com containers Linux e Docker Compose. No Windows/macOS, abra o Docker Desktop. Não é necessário instalar Python no computador do seu amigo. Os comandos completos estão em [comandos.txt](comandos.txt).

Dentro da pasta do projeto, construa a imagem e execute os seis experimentos:

```text
docker compose build
docker compose run --rm --pull never mnist
```

A imagem inclui Python, bibliotecas e MNIST. O build usa `dados/mnist.npz` quando disponível ou baixa o dataset durante a construção. Depois disso, o programa pode executar sem internet. As configurações são lidas do `configuracoes.json` da pasta do projeto; os resultados ficam em `resultados/` no computador, mesmo após a remoção do container.

Para levar a imagem pronta no pendrive:

```text
docker save -o ia-mlp-mnist-linux-amd64.tar ia-mlp-mnist:latest
```

Copie a pasta inteira, incluindo o TAR. No computador do seu amigo, com Docker já instalado e aberto, entre na pasta copiada e execute:

```text
docker load -i ia-mlp-mnist-linux-amd64.tar
docker compose run --rm --pull never mnist
```

A imagem usa Linux amd64, adequado a PCs Intel/AMD de 64 bits; em computadores ARM, depende da emulação oferecida pelo Docker. Se houver internet, também é possível construir a imagem diretamente no outro PC, sem transportar o TAR.

Após concluir o treinamento, demonstre o modelo salvo ou execute a suíte de testes:

```text
docker compose run --rm --pull never mnist demonstrar --indices 0 1 2 3 4 8 --sem-exibir
docker compose run --rm --pull never mnist testar
```

No Docker, a demonstração usa o terminal com `--sem-exibir`. Se a pasta `resultados/` copiada já contém uma execução concluída, é possível demonstrar sem treinar novamente. A busca do modelo aceita resultados copiados entre Windows, Linux e Docker. Abra os PNGs de treinamento diretamente pela pasta `resultados/`. Para alterar parâmetros, passe os mesmos argumentos da execução local depois de `mnist`. Alterações no código ou nas bibliotecas exigem reconstruir a imagem e exportar o TAR novamente.

### Instalação local com Python

Requer Python 3.10 ou superior. No PowerShell, dentro da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Os comandos abaixo usam o Python disponível no terminal. Para usar o ambiente virtual sem ativá-lo, substitua `python` por `.\.venv\Scripts\python.exe`.

O ambiente de validação utilizou Python 3.14.7, NumPy 2.4.4 e Matplotlib 3.11.1. As versões exatas de cada execução ficam em `configuracao.json`.

## Comandos principais

Treinar uma rede com uma camada intermediária de 128 neurônios:

```powershell
python main.py treinar --camadas 128 --taxa 0.01 --epocas 15 --lote 128
```

Treinar uma arquitetura com duas camadas:

```powershell
python main.py treinar --camadas 64 32 --taxa 0.01
```

Executar os **seis experimentos completos** definidos no trabalho:

```powershell
python main.py experimentos
```

Todos usam 15 épocas, mini-batches de 128, semente 42 e a mesma divisão de dados. A rede salva é a da última época; não há early stopping ou escolha de época pelo teste.

Modificar os parâmetros compartilhados:

```powershell
python main.py experimentos --epocas 20 --lote 64 --semente 42
```

Para mudar as arquiteturas e taxas de todas as execuções, edite uma cópia de `configuracoes.json` e informe seu caminho:

```powershell
python main.py experimentos --configuracoes configuracoes.json
```

As opções `--pasta-dados`, `--pasta-resultados` e `--threads` também são configuráveis. O padrão de uma thread evita o custo de usar muitas threads para matrizes pequenas. Para ajuda: `python main.py experimentos --help`.

## Demonstração de reconhecimento

Usar o melhor modelo da última execução com dados completos e abrir a imagem:

```powershell
python main.py demonstrar --indices 0
python main.py demonstrar --indices 0 1 2 3 4
python main.py demonstrar --quantidade 10 --semente 42
```

A demonstração mostra a classe prevista, a classe correta, acerto/erro e as dez probabilidades, sem salvar arquivos. Para mostrar as previsões apenas no terminal:

```powershell
python main.py demonstrar --indices 0 1 2 --sem-exibir
```

É possível escolher qualquer modelo salvo:

```powershell
python main.py demonstrar --modelo "resultados\PASTA_DA_EXECUCAO\execucao_3\modelo.npz" --indices 0 --sem-exibir
```

Substitua `PASTA_DA_EXECUCAO` pela pasta real. Os índices são os das 10.000 imagens originais de teste, começando em zero. Essas imagens não participam de treinamento ou seleção. A demonstração não treina a rede nem modifica seus parâmetros.

## Dataset e metodologia

Na primeira execução, `urllib.request` baixa o arquivo público [`mnist.npz`](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) para `dados/`. Depois, o programa usa a cópia local e funciona sem rede. O endereço hospedado pelo projeto TensorFlow serve apenas como fonte de dados; nenhuma dessas bibliotecas é importada.

As 60.000 imagens originais de treinamento são embaralhadas com `np.random.default_rng(42)`. As primeiras 5.000 da permutação ficam para validação e as 55.000 restantes para treinamento. A cada época, outro gerador com a mesma semente embaralha somente o treino. Como esse gerador é independente da inicialização dos pesos, as seis redes recebem a mesma ordem de mini-batches.

Cada imagem 28 × 28 vira um vetor de 784 valores `float32`, normalizados por 255. Os rótulos inteiros são convertidos em one-hot no cálculo da perda e dos gradientes. As camadas intermediárias usam ReLU e inicialização He; a saída tem 10 neurônios, inicialização com variância `1/fan_in` e Softmax. A previsão é `argmax`.

A perda é calculada dos logits com log-sum-exp, evitando logaritmo de zero. A derivada conjunta Softmax/entropia cruzada é `(probabilidades - one_hot) / tamanho_do_lote`. As camadas anteriores recebem o erro pela regra da cadeia e pela derivada da ReLU. A atualização é descida do gradiente sem momentum.

O programa treina todas as configurações, escolhe a maior **acurácia final de validação** e registra a escolha em `selecao_validacao.json`. Só depois carrega as 10.000 imagens de teste e avalia as configurações previamente definidas. Desempates usam menor perda de validação e, depois, menor número da execução. Não se deve alterar hiperparâmetros em função do teste.

## Resultados

Os seis experimentos completos foram executados e verificados em 14/10/2026. O modelo 128–64, escolhido pela validação (93,08%), obteve **94,08% no teste**, com 9.408 acertos e 592 erros. O fluxo levou 38,66 segundos neste computador. A [documentação para o artigo](DOCUMENTACAO_ARTIGO.md) contém as tabelas reais; a [pasta dessa execução](resultados/completo_20261014_162132_231359/) contém todos os arquivos. Os 15 testes passaram sem casos pulados.

Cada execução cria uma pasta diferente em `resultados/`, com prefixo `completo_`. `ultima_execucao_completa.json` informa o caminho da última execução. O treinamento usa 55.000 imagens de treino, 5.000 de validação e 10.000 de teste.

| Arquivo | Conteúdo |
|---|---|
| `resultados.json` | Arquitetura, taxa, épocas, tempo, perdas, acurácias, acertos e erros |
| `selecao_validacao.json` | Modelo escolhido antes da leitura do teste e critério de seleção |
| `configuracao.json` | Parâmetros, versões, plataforma, tempos e hashes dos dados e código |
| `execucao_N/historico.json` | Único histórico: perda e acurácia de treino/validação em cada época |
| `execucao_N/modelo.npz` | Pesos, bias, arquitetura e semente do modelo final |
| `perda.png`, `acuracia.png` | Curvas por época das configurações |
| `matriz_confusao.png` | Matriz 10 × 10 do melhor modelo |
| `acertos_por_classe.json` | Total, acertos, erros e acurácia por dígito |

Cada pasta `execucao_N` contém somente `historico.json` e `modelo.npz`. O modelo guarda os pesos da rede e é necessário para avaliar o teste e demonstrar o reconhecimento sem treinar novamente. As únicas figuras salvas são `perda.png`, `acuracia.png` e `matriz_confusao.png`.

As saídas para consulta são JSON e PNG. O programa não gera tabelas ou relatórios adicionais, nem cópias do melhor modelo. A demonstração localiza o modelo diretamente em `execucao_N/modelo.npz` usando a última execução e seu identificador selecionado. A semente em `configuracao.json` permite reconstruir a divisão de treino e validação com `carregar_treino_validacao`.

Nos JSONs de resultados, percentuais têm o sufixo `_percentual`. Nos históricos, acurácias são proporções entre 0 e 1. O tempo de treinamento inclui avaliação das métricas em cada época; não inclui download, avaliação inicial, salvamento ou figuras. `tempo_total_segundos` mede o fluxo completo após seu início, incluindo preparação e figuras.

## Testes

```powershell
python main.py verificar-gradientes
python main.py testar
```

Não é necessário instalar pytest. A suíte usa `unittest`, da biblioteca padrão, e verifica gradientes por diferenças finitas, estabilidade numérica, dimensões, aprendizado das dez classes, reprodução, partições disjuntas, ausência de leitura do teste no preparo do treino, seleção pela validação, serialização, geração das figuras e demonstração. A integração usa amostras menores do MNIST em uma pasta temporária, removida automaticamente, sem gravar saídas em `resultados/`. Os testes do dataset e da integração são pulados quando o MNIST ainda não está em `dados/`; um teste pulado não equivale a validação executada.

## Organização

```text
main.py                    comandos do programa
mlp.py                     rede e Backpropagation manual
dados.py                   download e preparação do MNIST
experimentos.py            treinamento, seleção e resultados
graficos.py                figuras dos resultados
demonstracao.py            reconhecimento com modelo salvo
configuracoes.json         exemplo editável dos seis experimentos
requirements.txt           NumPy e Matplotlib
Dockerfile                 imagem com Python, bibliotecas e MNIST
compose.yaml               execução Docker e resultados persistentes
comandos.txt               comandos para preparar e executar na faculdade
tests/                     testes matemáticos e de integração
DOCUMENTACAO_ARTIGO.md     metodologia e resultados para o artigo
dados/                     cópia local do MNIST
resultados/                saídas reais de cada execução
```

As pastas de dados e resultados estão no `.gitignore` para evitar versionar arquivos grandes. Para entregar o trabalho com as evidências, inclua a pasta da execução completa na cópia enviada ou no arquivo compactado.

## Limitações

Os modelos reconhecem imagens do MNIST; não há pré-processamento de fotografias ou desenhos externos. Uma execução por configuração não mede variação entre sementes. Tempos e diferenças numéricas podem variar entre computadores e versões de bibliotecas. As probabilidades Softmax não foram calibradas. O artigo no modelo SBC deve ser escrito posteriormente com base na documentação e nos resultados reais.
