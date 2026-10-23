from pathlib import Path
from time import perf_counter

import numpy as np

# Marca o rotulo cero com 0
def one_hot(rotulos, numero_classes=10, dtype=np.float32):
    """Converte, por exemplo, a classe 2 em [0, 0, 1, ...]."""
    rotulos = np.asarray(rotulos)
    if rotulos.ndim != 1 or not np.issubdtype(rotulos.dtype, np.integer):
        raise ValueError("Os rótulos devem ser um vetor de números inteiros.")
    if np.any(rotulos < 0) or np.any(rotulos >= numero_classes):
        raise ValueError("Há rótulos fora do intervalo de classes.")
    return np.eye(numero_classes, dtype=dtype)[rotulos]

# Probabilidade de cada numero
def softmax(logits):
    # Subtrair o maior logit evita overflow e não altera as probabilidades.
    deslocados = logits - np.max(logits, axis=1, keepdims=True) # evita overflow
    exponenciais = np.exp(deslocados)
    return exponenciais / np.sum(exponenciais, axis=1, keepdims=True)

def entropia_cruzada(logits, alvos):
    """Calcula a perda diretamente dos logits, sem log de probabilidades zero."""
    deslocados = logits - np.max(logits, axis=1, keepdims=True)
    # log-sum-exp estável: log(Softmax(z)) = z - log(sum(exp(z))).
    log_probabilidades = deslocados - np.log(
        np.sum(np.exp(deslocados), axis=1, keepdims=True)
    )
    return float(-np.mean(np.sum(alvos * log_probabilidades, axis=1)))


class MLP:
    def __init__(self, camadas, semente=42, dtype=np.float32):
        if len(camadas) < 3 or any(
            not isinstance(n, (int, np.integer)) or n <= 0 for n in camadas
        ):
            raise ValueError("Use entrada, ao menos uma camada intermediária e saída positivas.")
        self.camadas = [int(n) for n in camadas]
        self.semente = int(semente)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("A rede utiliza float32 ou float64.")
        gerador = np.random.default_rng(semente)
        self.pesos = []
        self.bias = []
        for i, (entrada, saida) in enumerate(zip(camadas[:-1], camadas[1:])):
            # He nas camadas ReLU; variância 1/fan_in na saída linear.
            escala = np.sqrt((2.0 if i < len(camadas) - 2 else 1.0) / entrada)
            self.pesos.append(
                (gerador.standard_normal((entrada, saida)) * escala).astype(self.dtype)
            )
            self.bias.append(np.zeros((1, saida), dtype=self.dtype))

    def _validar_entrada(self, x):
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 2 or x.shape[1] != self.camadas[0] or len(x) == 0:
            raise ValueError(f"A entrada deve ter formato (N, {self.camadas[0]}), com N > 0.")
        if not np.all(np.isfinite(x)):
            raise ValueError("A entrada contém valores não finitos.")
        return x

    # Realiza a propagação para frente 128 -> 64
    def _forward(self, x):
        ativacoes = [x]
        pre_ativacoes = []
        for i, (peso, bias) in enumerate(zip(self.pesos, self.bias)):
            z = ativacoes[-1] @ peso + bias
            pre_ativacoes.append(z)
            # ReLU(z) = max(0, z); na última camada aplicamos Softmax.
            a = np.maximum(0, z) if i < len(self.pesos) - 1 else softmax(z)
            ativacoes.append(a)
        return ativacoes, pre_ativacoes

    # Retorna a probabilidade final de ser um digito
    def forward(self, x):
        ativacoes, _ = self._forward(self._validar_entrada(x))
        return ativacoes[-1]

    # Retorna a penalização
    def calcular_perda(self, x, y):
        x = self._validar_entrada(x)
        alvos = one_hot(y, self.camadas[-1], self.dtype)
        if len(alvos) != len(x):
            raise ValueError("Imagens e rótulos devem ter o mesmo tamanho.")
        _, pre_ativacoes = self._forward(x)
        return entropia_cruzada(pre_ativacoes[-1], alvos)

    # Backpropagation
    def gradientes(self, x, y):
        """Retorna perda, dL/dW e dL/db sem modificar os parâmetros."""
        x = self._validar_entrada(x)
        alvos = one_hot(y, self.camadas[-1], self.dtype)
        if len(alvos) != len(x):
            raise ValueError("Imagens e rótulos devem ter o mesmo tamanho.")

        # Forward e perda
        ativacoes, pre_ativacoes = self._forward(x)
        perda = entropia_cruzada(pre_ativacoes[-1], alvos)

        # indica como a perda muda se mdudarmos um parametro
        grad_pesos = [None] * len(self.pesos)
        grad_bias = [None] * len(self.bias)

        # Softmax + entropia cruzada: dL/dZ = (probabilidade - alvo) / N.
        delta = (ativacoes[-1] - alvos) / len(x)

        # percorre camadas de tras para frente
        for i in range(len(self.pesos) - 1, -1, -1):
            # Regra da cadeia para Z = A @ W + b.
            grad_pesos[i] = ativacoes[i].T @ delta #transposta e multiplica
            grad_bias[i] = np.sum(delta, axis=0, keepdims=True)

            # Propaga o gradiente para a camada anterior
            if i > 0:
                # ReLU'(z) = 1 para z > 0 e 0 para z <= 0.
                delta = (delta @ self.pesos[i].T) * (pre_ativacoes[i - 1] > 0)
        return perda, grad_pesos, grad_bias

    #  Atualiza pesos e bias
    def atualizar(self, grad_pesos, grad_bias, taxa_aprendizado):
        # Descida do gradiente: parâmetro novo = antigo - taxa * gradiente.
        for i in range(len(self.pesos)):
            self.pesos[i] -= taxa_aprendizado * grad_pesos[i]
            self.bias[i] -= taxa_aprendizado * grad_bias[i]

    # Processa imagens e retorna as probabilidades
    def prever_probabilidades(self, x, tamanho_lote=1024):
        x = self._validar_entrada(x)
        if tamanho_lote <= 0:
            raise ValueError("O tamanho do lote deve ser positivo.")
        return np.concatenate(
            [self.forward(x[i:i + tamanho_lote]) for i in range(0, len(x), tamanho_lote)]
        )

    # Escolhe a maior probabilidade
    def prever(self, x, tamanho_lote=1024):
        return np.argmax(self.prever_probabilidades(x, tamanho_lote), axis=1)

    # retorna perda média, a acurácia, a quantidade de acertos e a quantidade de erros.
    def avaliar(self, x, y, tamanho_lote=1024):
        x = self._validar_entrada(x)
        y = np.asarray(y)
        one_hot(y, self.camadas[-1], self.dtype)
        if len(x) != len(y) or tamanho_lote <= 0:
            raise ValueError("Dados incompatíveis ou tamanho de lote inválido.")
        soma_perdas, acertos = 0.0, 0
        for inicio in range(0, len(x), tamanho_lote):
            lote_x = x[inicio:inicio + tamanho_lote]
            lote_y = y[inicio:inicio + tamanho_lote]
            ativacoes, pre_ativacoes = self._forward(lote_x)
            soma_perdas += len(lote_x) * entropia_cruzada(
                pre_ativacoes[-1], one_hot(lote_y, self.camadas[-1], self.dtype)
            )
            acertos += int(np.sum(np.argmax(ativacoes[-1], axis=1) == lote_y))
        return {"perda": soma_perdas / len(x), "acuracia": acertos / len(x),
                "acertos": acertos, "erros": len(x) - acertos}

    def treinar(self, x_treino, y_treino, x_validacao, y_validacao,
                epocas=15, tamanho_lote=128, taxa_aprendizado=0.01,
                semente=42, verbose=True):
        if epocas <= 0 or tamanho_lote <= 0 or not np.isfinite(taxa_aprendizado) or taxa_aprendizado <= 0:
            raise ValueError("Épocas, tamanho do lote e taxa devem ser positivos.")
        x_treino = self._validar_entrada(x_treino)
        x_validacao = self._validar_entrada(x_validacao)
        # Valida os rótulos antes de qualquer atualização.
        for x, y in ((x_treino, y_treino), (x_validacao, y_validacao)):
            if len(x) != len(y):
                raise ValueError("Imagens e rótulos devem ter o mesmo tamanho.")
            one_hot(y, self.camadas[-1], self.dtype)
        y_treino = np.asarray(y_treino)
        gerador = np.random.default_rng(semente)
        historico = []
        for epoca in range(1, epocas + 1):
            inicio = perf_counter()
            # Gerador separado dos pesos: mesma ordem de lotes nas seis redes.
            ordem = gerador.permutation(len(x_treino))
            # para cada lote Executa o forward, Backpropagation
            for posicao in range(0, len(ordem), tamanho_lote):
                indices = ordem[posicao:posicao + tamanho_lote]
                perda, grad_pesos, grad_bias = self.gradientes(x_treino[indices], y_treino[indices])
                if not np.isfinite(perda) or any(
                    not np.all(np.isfinite(g)) for g in grad_pesos + grad_bias
                ):
                    raise FloatingPointError("Treinamento divergiu. Verifique a taxa de aprendizado.")
                self.atualizar(grad_pesos, grad_bias, taxa_aprendizado)
            # Avalia resultado final da epoca
            treino = self.avaliar(x_treino, y_treino)
            validacao = self.avaliar(x_validacao, y_validacao)
            registro = {"epoca": epoca, "perda_treino": treino["perda"],
                        "perda_validacao": validacao["perda"],
                        "acuracia_treino": treino["acuracia"],
                        "acuracia_validacao": validacao["acuracia"],
                        "tempo_epoca_segundos": perf_counter() - inicio}
            if not all(np.isfinite(v) for v in registro.values()):
                raise FloatingPointError("Foram obtidas métricas não finitas.")
            historico.append(registro)
            if verbose:
                print(f"Época {epoca:02d}/{epocas}: perda {treino['perda']:.4f} / "
                      f"{validacao['perda']:.4f}; acurácia {treino['acuracia']:.2%} / "
                      f"{validacao['acuracia']:.2%}; {registro['tempo_epoca_segundos']:.2f}s",
                      flush=True)
        return historico

    # grava a arquitetura, a semente, os pesos e os bias da rede em um arquivo compactado .npz
    def salvar(self, caminho):
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        conteudo = {"versao": np.array(1), "camadas": np.array(self.camadas),
                    "semente": np.array(self.semente)}
        for i, (peso, bias) in enumerate(zip(self.pesos, self.bias)):
            conteudo[f"peso_{i}"] = peso
            conteudo[f"bias_{i}"] = bias
        # Usar um arquivo aberto evita a adição silenciosa da extensão .npz.
        with caminho.open("wb") as arquivo:
            np.savez_compressed(arquivo, **conteudo)

    # lê esse arquivo .npz
    @classmethod
    def carregar(cls, caminho):
        with np.load(caminho, allow_pickle=False) as arquivo:
            if int(arquivo["versao"]) != 1:
                raise ValueError("Versão de modelo não suportada.")
            modelo = cls(arquivo["camadas"].tolist(), int(arquivo["semente"]),
                         dtype=arquivo["peso_0"].dtype)
            for i in range(len(modelo.pesos)):
                peso, bias = arquivo[f"peso_{i}"], arquivo[f"bias_{i}"]
                if peso.shape != modelo.pesos[i].shape or bias.shape != modelo.bias[i].shape:
                    raise ValueError("Dimensões inválidas no arquivo de modelo.")
                if not np.all(np.isfinite(peso)) or not np.all(np.isfinite(bias)):
                    raise ValueError("O modelo contém parâmetros não finitos.")
                modelo.pesos[i] = peso.astype(modelo.dtype)
                modelo.bias[i] = bias.astype(modelo.dtype)
        return modelo

# verifica se os gradientes calculados pelo Backpropagation estão corretos
def verificar_gradientes(epsilon=1e-5):
    """Compara TODOS os pesos e bias de uma rede pequena por diferenças centrais."""
    gerador = np.random.default_rng(7)
    modelo = MLP([4, 5, 4, 3], semente=7, dtype=np.float64)
    x = gerador.normal(size=(5, 4))
    y = np.array([0, 1, 2, 1, 0])
    _, grad_pesos, grad_bias = modelo.gradientes(x, y)
    maior_erro_absoluto, maior_erro_relativo, quantidade = 0.0, 0.0, 0
    for parametro, gradiente in zip(modelo.pesos + modelo.bias, grad_pesos + grad_bias):
        numerico = np.zeros_like(parametro)
        for indice in np.ndindex(parametro.shape):
            original = parametro[indice]
            parametro[indice] = original + epsilon
            perda_mais = modelo.calcular_perda(x, y)
            parametro[indice] = original - epsilon
            perda_menos = modelo.calcular_perda(x, y)
            parametro[indice] = original
            numerico[indice] = (perda_mais - perda_menos) / (2 * epsilon)
        diferenca = np.abs(numerico - gradiente)
        relativo = diferenca / np.maximum(1e-8, np.abs(numerico) + np.abs(gradiente))
        maior_erro_absoluto = max(maior_erro_absoluto, float(diferenca.max()))
        maior_erro_relativo = max(maior_erro_relativo, float(relativo.max()))
        quantidade += parametro.size
        np.testing.assert_allclose(gradiente, numerico, atol=1e-7, rtol=1e-5)
    return {"parametros_verificados": quantidade, "epsilon": epsilon,
            "maior_erro_absoluto": maior_erro_absoluto,
            "maior_erro_relativo": maior_erro_relativo, "aprovado": True}
