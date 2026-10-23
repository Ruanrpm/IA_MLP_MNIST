# Documentação técnica para o artigo

Este documento descreve a implementação e fornece evidências para a redação posterior de um artigo no modelo SBC. Não constitui o artigo completo. A fonte verificável dos números é a pasta de resultados identificada na seção de resultados.

## 1. Problema e objetivo

O problema consiste em classificar uma imagem de um dígito manuscrito em uma das dez classes, de 0 a 9. O objetivo é implementar uma rede neural Multilayer Perceptron (MLP) e seu treinamento por Backpropagation, comparando quantidade de camadas intermediárias, quantidade de neurônios e taxa de aprendizado. A implementação prioriza clareza matemática e organização compatível com um trabalho de graduação.

Não se utilizam modelos pré-treinados, bibliotecas de treinamento ou diferenciação automática. NumPy realiza operações matriciais e Matplotlib gera figuras; as demais bibliotecas são da distribuição padrão do Python.

## 2. Dataset e fonte

O MNIST contém imagens em tons de cinza de 28 × 28 pixels e rótulos de dígitos de 0 a 9. A distribuição utilizada possui 60.000 imagens originais de treinamento e 10.000 de teste, conforme a [documentação do conjunto MNIST](https://keras.io/api/datasets/mnist/).

O download é realizado diretamente por `urllib.request` do [arquivo público mnist.npz](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz). NumPy abre o arquivo com `allow_pickle=False`. A origem do arquivo não implica uso de TensorFlow ou Keras na solução. O dataset é armazenado em `dados/mnist.npz` para reutilização local.

Referência histórica: LeCun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. *Gradient-Based Learning Applied to Document Recognition*. Proceedings of the IEEE, 86(11), p. 2278–2324, 1998. [Publicação original e DOI](https://doi.org/10.1109/5.726791). O presente trabalho usa uma MLP; não reproduz a arquitetura convolucional desse artigo.

## 3. Divisão dos dados

| Conjunto | Quantidade na execução definitiva | Uso |
|---|---|---|
| Treinamento | 55.000 | Cálculo dos gradientes e atualização dos parâmetros |
| Validação | 5.000 | Comparação e seleção das configurações |
| Teste | 10.000 | Avaliação final, após a escolha pela validação |

Uma permutação reproduzível das 60.000 imagens originais de treinamento é produzida com `np.random.default_rng(42)`. As primeiras 5.000 posições compõem a validação e as demais 55.000 o treinamento. Os conjuntos são disjuntos. A semente e os hashes dos índices são registrados em `configuracao.json`; os índices podem ser reconstruídos pela mesma permutação. As dez mil imagens originais de teste não integram essa permutação.

`carregar_treino_validacao` não acessa as chaves `x_test` ou `y_test`. `carregar_teste` é chamado pela rotina experimental somente após todas as redes terem sido treinadas e a seleção ter sido registrada. A validação nunca é utilizada para calcular gradientes.

Os testes de integração usam amostras de 2.000/500/500 imagens em pastas temporárias removidas automaticamente. O treinamento pelos comandos do programa utiliza os conjuntos completos.

## 4. Pré-processamento

1. Verificar dimensões das imagens e a faixa dos rótulos.
2. Achatar cada imagem de 28 × 28 para um vetor de 784 posições, preservando a ordem dos pixels.
3. Converter `uint8` para `float32` e dividir cada pixel por 255, obtendo valores entre 0 e 1.
4. Converter os rótulos inteiros para vetores one-hot durante o cálculo da perda e dos gradientes. O rótulo 3 corresponde ao vetor cujo quarto elemento vale 1 e os demais valem 0.

Não se aplicam aumento de dados, centralização adicional, transformação de imagens ou extração manual de características. A normalização depende de uma constante e não estima estatísticas a partir do teste.

## 5. Arquitetura e inicialização

A entrada tem 784 valores; cada camada intermediária é totalmente conectada e usa ReLU. A saída contém dez valores transformados em probabilidades por Softmax. O programa permite uma ou mais camadas intermediárias com tamanhos configuráveis.

Para uma camada com `n_entrada` e `n_saida` neurônios, a matriz de pesos tem formato `(n_entrada, n_saida)` e o bias tem formato `(1, n_saida)`. Nas camadas ReLU, os pesos são sorteados de uma normal de média zero e desvio padrão `sqrt(2/n_entrada)`, a inicialização He. Na camada de saída linear, utiliza-se `sqrt(1/n_entrada)`. Todos os bias começam em zero.

A mesma semente é utilizada para cada inicialização, mas matrizes de formatos diferentes não representam redes com pesos idênticos. A ordem dos mini-batches usa um gerador independente, garantindo o mesmo embaralhamento em todas as arquiteturas.

## 6. Propagação para frente

Para um lote com `m` imagens, seja `A^(0) = X`, de formato `(m, 784)`. Em cada camada `l`:

```text
Z^(l) = A^(l-1) W^(l) + b^(l)
A^(l) = max(0, Z^(l))                 nas camadas intermediárias
P = Softmax(Z^(L))                    na saída
```

O bias é somado a todas as linhas por broadcasting. A Softmax é calculada por imagem:

```text
s_j = z_j - max_k(z_k)
p_j = exp(s_j) / sum_k(exp(s_k))
```

A subtração do maior logit evita overflow. A previsão é o índice de maior probabilidade, `argmax(P, axis=1)`, correspondendo a uma classe entre 0 e 9.

## 7. Função de perda

A Categorical Cross-Entropy média de um lote é:

```text
L = -(1/m) sum_i sum_j Y_ij log(P_ij)
```

O código utiliza a forma equivalente diretamente dos logits:

```text
log(P_ij) = s_ij - log(sum_k(exp(s_ik)))
```

Essa forma evita calcular o logaritmo de uma probabilidade arredondada para zero. O teste com logits de magnitude 1.000 verifica que tanto a Softmax quanto a perda permanecem finitas.

## 8. Backpropagation e regra da cadeia

A derivada conjunta da Softmax e da entropia cruzada na saída é:

```text
Delta^(L) = (P - Y) / m
```

Percorrendo as camadas da saída para a entrada:

```text
dL/dW^(l) = (A^(l-1))^T Delta^(l)
dL/db^(l) = sum_i Delta_i^(l)
Delta^(l-1) = [Delta^(l) (W^(l))^T] * [Z^(l-1) > 0]
```

O último produto é elemento a elemento. O indicador `[Z > 0]` representa a derivada da ReLU: 1 nos valores positivos e 0 nos demais. Em zero, adota-se derivada 0. A divisão pelo tamanho real do lote é feita uma única vez na saída, inclusive quando o último lote tem menos de 128 exemplos.

Todos os gradientes são calculados antes da atualização dos parâmetros, preservando os pesos usados no forward. A função `MLP.gradientes` permite inspecionar a perda e cada matriz de derivadas sem modificar o modelo.

## 9. Atualização e treinamento

Para taxa de aprendizado `eta`:

```text
W^(l) = W^(l) - eta * dL/dW^(l)
b^(l) = b^(l) - eta * dL/db^(l)
```

Utiliza-se descida do gradiente em mini-batches, sem momentum, regularização, ajuste automático da taxa ou parada antecipada. Uma época percorre todas as imagens de treino uma vez. No final de cada época, perda e acurácia são recalculadas em todos os dados de treino e validação com os mesmos parâmetros atualizados.

São utilizados 15 épocas, lotes de 128 e semente 42. O modelo salvo corresponde à última época. A implementação detecta perdas ou gradientes não finitos e interrompe um treinamento divergente.

## 10. Configurações experimentais e seleção

| Execução | Camadas intermediárias | Neurônios | Taxa de aprendizado |
|---|---|---|---|
| 1 | 1 | 32 | 0,01 |
| 2 | 1 | 64 | 0,01 |
| 3 | 1 | 128 | 0,01 |
| 4 | 2 | 64 e 32 | 0,01 |
| 5 | 2 | 64 e 32 | 0,001 |
| 6 | 2 | 128 e 64 | 0,01 |

As seis configurações são definidas antes da execução. Todas compartilham dados, ordem de mini-batches, semente, ativações, função de perda, número de épocas e tamanho de lote.

A escolha utiliza a maior acurácia de validação na última época. Em caso de empate, utiliza-se menor perda de validação e, se necessário, menor identificador da execução. `selecao_validacao.json` registra a escolha antes de carregar o teste. As seis configurações são avaliadas no teste ao final porque foram previamente definidas. Não foram feitos ajustes de hiperparâmetros posteriores aos resultados de teste.

## 11. Métricas e rastreabilidade

```text
acurácia = quantidade de acertos / quantidade de imagens
erros = quantidade de imagens - acertos
acurácia percentual = 100 * acurácia
```

A matriz de confusão é salva somente em `matriz_confusao.png`, com linhas para classes reais e colunas para classes previstas. Sua diagonal conta os acertos por classe. `acertos_por_classe.json` inclui suporte, acertos, erros e acurácia dentro de cada classe; uma classe com menos exemplos não deve ser comparada às demais apenas pela quantidade bruta de acertos.

Os resultados são salvos em `resultados.json`. Cada pasta `execucao_N` contém apenas o histórico por época em `historico.json` e os pesos da rede em `modelo.npz`. O modelo selecionado é utilizado diretamente dessa pasta, sem criar uma cópia. `configuracao.json` registra versões, plataforma, sementes, tamanhos, hashes SHA-256 do dataset, dos índices e dos arquivos principais do programa. Esses registros permitem localizar os dados e o código associados à execução.

O tempo de treinamento inclui os mini-batches e a avaliação de treino/validação por época. Exclui download, avaliação inicial, salvamento e geração das figuras. O tempo total do fluxo é registrado separadamente. Tempos não são determinísticos, mesmo quando pesos e métricas se reproduzem.

## 12. Resultados reais

<!-- RESULTADOS_REAIS_INICIO -->
Execução realizada em **14/10/2026**, com o conjunto completo, 15 épocas, lotes de 128 e semente 42. Ambiente: Python 3.14.7, NumPy 2.4.4 e Matplotlib 3.11.1, Windows 11, uma thread nas operações matriciais. O fluxo completo levou **38,66 segundos**. Os tempos variam entre execuções e computadores.

Pasta de referência: [`resultados/completo_20261014_162132_231359/`](resultados/completo_20261014_162132_231359/). Consulte os [resultados originais](resultados/completo_20261014_162132_231359/resultados.json), a [seleção pela validação](resultados/completo_20261014_162132_231359/selecao_validacao.json) e a [verificação histórica dos arquivos](resultados/completo_20261014_162132_231359/verificacao.json). Esta seção é um registro dessa execução; novas execuções geram seus próprios JSONs e figuras sem sobrescrever este registro.

| Execução | Camadas | Neurônios | Taxa | Épocas | Tempo (s) | Validação (%) | Acertos no teste | Erros no teste | Teste (%) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 32 | 0.01 | 15 | 3.53 | 91.18 | 9225 | 775 | 92.25 |
| 2 | 1 | 64 | 0.01 | 15 | 4.93 | 91.20 | 9231 | 769 | 92.31 |
| 3 | 1 | 128 | 0.01 | 15 | 7.54 | 91.14 | 9261 | 739 | 92.61 |
| 4 | 2 | 64-32 | 0.01 | 15 | 5.30 | 92.52 | 9350 | 650 | 93.50 |
| 5 | 2 | 64-32 | 0.001 | 15 | 5.25 | 85.06 | 8652 | 1348 | 86.52 |
| 6 | 2 | 128-64 | 0.01 | 15 | 8.57 | 93.08 | 9408 | 592 | 94.08 |

### Perdas e acurácia de treinamento na última época

| Execução | Perda inicial de treino | Perda final de treino | Perda final de validação | Treinamento (%) |
|---|---|---|---|---|
| 1 | 2.3640 | 0.2801 | 0.3045 | 92.09 |
| 2 | 2.2979 | 0.2785 | 0.3060 | 92.14 |
| 3 | 2.3660 | 0.2702 | 0.2955 | 92.35 |
| 4 | 2.3223 | 0.2249 | 0.2503 | 93.56 |
| 5 | 2.3223 | 0.5501 | 0.5732 | 85.91 |
| 6 | 2.3827 | 0.2053 | 0.2282 | 94.21 |

A **execução 6** foi selecionada com 93,08% na validação, antes da leitura do teste. Sua acurácia final de teste foi 94,08%, com 9.408 acertos e 592 erros. Seu modelo está em `execucao_6/modelo.npz`. As perdas diminuíram em todas as seis redes. A diferença entre acurácia de treino e validação do modelo escolhido foi 1,13 ponto percentual; esse valor sozinho não determina se há sobreajuste.

### Acertos por classe do modelo escolhido

| Dígito | Imagens de teste | Acertos | Erros | Acurácia na classe (%) |
|---|---|---|---|---|
| 0 | 980 | 965 | 15 | 98.47 |
| 1 | 1135 | 1115 | 20 | 98.24 |
| 2 | 1032 | 962 | 70 | 93.22 |
| 3 | 1010 | 941 | 69 | 93.17 |
| 4 | 982 | 926 | 56 | 94.30 |
| 5 | 892 | 814 | 78 | 91.26 |
| 6 | 958 | 907 | 51 | 94.68 |
| 7 | 1028 | 963 | 65 | 93.68 |
| 8 | 974 | 888 | 86 | 91.17 |
| 9 | 1009 | 927 | 82 | 91.87 |

Os dígitos 0 e 1 apresentaram as maiores acurácias por classe, e o 8 apresentou a menor. Na matriz de confusão, ocorreram 28 previsões de 9 para imagens de 4 e 28 previsões de 4 para imagens de 9. Também ocorreram 25 previsões de 3 para imagens de 5. As direções desses erros são distintas e estão registradas separadamente.

### Discussão dos resultados observados

Nas redes com uma camada, aumentar a largura de 32 para 64 ou 128 neurônios não gerou crescimento monotônico da acurácia de validação: os resultados foram 91,18%, 91,20% e 91,14%. Portanto, esta execução não sustenta a afirmação de que mais neurônios sempre produzem melhor classificação.

Na arquitetura 64–32, a taxa 0,01 apresentou acurácia de validação 7,46 pontos percentuais acima da taxa 0,001 após as mesmas 15 épocas. As curvas da taxa menor ainda mostram redução da perda, compatível com progresso mais lento nesse orçamento de treinamento. Não foi realizado um treinamento mais longo para comparar as taxas após convergência.

A rede 64–32 superou a rede de uma camada com 64 neurônios em 1,32 ponto percentual na validação. A rede 128–64 superou a de uma camada com 128 neurônios em 1,94 ponto percentual. A execução 6 teve o maior tempo de treinamento, 8,57 segundos, e o maior desempenho de validação entre as configurações avaliadas. Essas diferenças descrevem esta semente e este particionamento; não foram submetidas a teste estatístico.

Figuras mantidas para a apresentação: [perda](resultados/completo_20261014_162132_231359/perda.png), [acurácia por época](resultados/completo_20261014_162132_231359/acuracia.png) e [matriz de confusão](resultados/completo_20261014_162132_231359/matriz_confusao.png).

### Evidências de validação

Os **15 testes passaram sem casos pulados**, incluindo a integração com amostras do MNIST real. A verificação dos 64 gradientes apresentou erro absoluto máximo de `1.7220003201146028e-11` e erro relativo máximo de `6.056829867534727e-10`.

Na validação original, os seis modelos completos foram carregados novamente, e suas perdas e acurácias de treino, validação e teste foram recalculadas e confrontadas com os arquivos salvos. As métricas concordaram; as partições e hashes foram conferidos; todos os modelos produziram previsões das dez classes no teste. A matriz de confusão e os acertos por classe foram reconstruídos a partir das previsões. As figuras foram lidas com sucesso e as principais também foram inspecionadas visualmente. O arquivo `verificacao.json` preserva esse registro histórico, anterior à retirada das saídas auxiliares.

A demonstração foi executada sem abrir janela para os índices 0, 1, 2, 3 e 4, reconhecendo corretamente 7, 2, 1, 0 e 4. Esse pequeno exemplo ilustra o uso do modelo e não substitui a avaliação das 10.000 imagens. A demonstração atual exibe as previsões na janela ou apenas no terminal com `--sem-exibir`, sem salvar arquivos. A janela não foi aberta durante a validação automática.
<!-- RESULTADOS_REAIS_FIM -->

## 13. Interpretação das figuras

- **Perda:** redução da entropia cruzada indica que as probabilidades se aproximam dos rótulos. Divergência entre curvas de treino e validação pode indicar sobreajuste; a interpretação deve considerar as curvas reais.
- **Acurácia por época:** mostra a evolução dos acertos. Variações pequenas de uma época para outra podem ocorrer por causa da ordem dos lotes e das atualizações.
- **Matriz de confusão:** a diagonal reúne classificações corretas. Células fora da diagonal mostram quais pares de dígitos foram confundidos e em qual direção.

## 14. Discussão e limites da comparação

A quantidade de neurônios altera a capacidade e o custo de cálculo da rede. Uma segunda camada adiciona transformações não lineares, mas não garante maior acurácia. As execuções 4 e 5 isolam a taxa de aprendizado na arquitetura 64–32. As demais comparações permitem examinar largura e profundidade dentro das configurações definidas.

O número fixo de épocas faz parte do protocolo. Uma taxa menor pode precisar de mais épocas; um resultado inferior após 15 épocas não prova que ela terá desempenho inferior após convergência. A relação entre arquitetura e qualidade deve ser discutida a partir da validação, dos tempos e das curvas observadas, sem generalizar a partir de uma única execução.

Não há repetição com múltiplas sementes, intervalos de confiança ou teste estatístico. Diferenças pequenas de acurácia não demonstram superioridade estatística. O teste foi observado somente depois da seleção, e novas mudanças motivadas por esses números exigiriam rever o protocolo de avaliação.

## 15. Verificações e dificuldades

A verificação matemática compara todos os 64 pesos e bias de uma rede 4–5–4–3 com diferenças finitas centrais, em `float64`, usando `epsilon = 1e-5`. A tolerância é `atol = 1e-7` e `rtol = 1e-5`. O treinamento normal usa `float32`.

A suíte verifica estabilidade numérica, dimensões, aprendizado de dez padrões sintéticos, reprodução dos parâmetros e métricas, salvamento/carregamento, divisão do MNIST e seleção independente do teste. O teste de integração executa as seis configurações com amostras menores em uma pasta temporária, verifica a existência da seleção antes da leitura do teste, lê os PNGs, confere contagens e executa a demonstração. O conjunto sintético serve somente para verificar o algoritmo; não é apresentado como resultado de reconhecimento do MNIST.

Neste ambiente, o primeiro download foi bloqueado pela restrição de rede e foi concluído com a execução autorizada fora dessa restrição. Depois, o dataset local permitiu executar os experimentos sem rede. Um problema de codificação do terminal do Windows foi corrigido configurando os fluxos de saída em UTF-8.

Outras limitações: reconhecimento restrito ao formato MNIST; ausência de suporte a imagens externas; probabilidades não calibradas; diferença possível entre versões e implementações de BLAS; a janela interativa da demonstração não é aberta na validação automática.

## 16. Como usar no artigo

Utilize a metodologia descrita acima, a tabela da execução completa e as figuras de sua pasta. Apresente a semente e os parâmetros compartilhados. Identifique explicitamente que o melhor modelo foi escolhido na validação. Discuta os erros por classe e as limitações, sem inventar dados, repetir o teste para escolher parâmetros ou transformar uma única execução em evidência de superioridade estatística. A redação e a formatação final no modelo SBC permanecem para a etapa de elaboração do artigo.
