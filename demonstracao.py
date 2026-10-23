import json
import math
from pathlib import Path, PureWindowsPath

import numpy as np

from dados import PASTA_DADOS, carregar_teste
from mlp import MLP


def localizar_modelo(pasta_resultados):
    pasta_resultados = Path(pasta_resultados)
    indice = pasta_resultados / "ultima_execucao_completa.json"
    if indice.exists():
        referencia = json.loads(indice.read_text(encoding="utf-8"))
        # Prioriza a copia local quando os resultados mudam de PC ou entram no Docker.
        # PureWindowsPath reconhece tanto barras do Linux quanto barras do Windows.
        pasta_local = pasta_resultados / PureWindowsPath(referencia["pasta"]).name
        for pasta in (pasta_local, Path(referencia["pasta"])):
            modelo = pasta / f"execucao_{referencia['melhor_execucao']}" / "modelo.npz"
            if modelo.exists():
                return modelo
    raise FileNotFoundError("Nenhum modelo encontrado. Execute o treinamento ou informe --modelo.")


def demonstrar(caminho_modelo, indices=None, quantidade=None, semente=42,
               pasta_dados=PASTA_DADOS, exibir=True):
    modelo = MLP.carregar(caminho_modelo)
    if modelo.camadas[0] != 784 or modelo.camadas[-1] != 10:
        raise ValueError("A demonstração requer uma rede com 784 entradas e 10 saídas.")
    x_teste, y_teste = carregar_teste(pasta_dados)
    if quantidade is not None:
        if quantidade <= 0 or quantidade > len(y_teste):
            raise ValueError("A quantidade deve estar entre 1 e 10.000.")
        indices = np.random.default_rng(semente).choice(len(y_teste), quantidade, replace=False).tolist()
    elif indices is None:
        indices = [0]
    if not indices or any(i < 0 or i >= len(y_teste) for i in indices):
        raise ValueError("Os índices devem estar entre 0 e 9.999.")
    probabilidades = modelo.prever_probabilidades(x_teste[indices])
    previsoes = np.argmax(probabilidades, axis=1)
    registros = []
    for indice, previsto, probs in zip(indices, previsoes, probabilidades):
        correto = int(y_teste[indice])
        acerto = int(previsto) == correto
        print(f"Imagem {indice}: previsto {previsto}; classe correta {correto}; "
              f"previsão {'CORRETA' if acerto else 'INCORRETA'}.")
        print("Probabilidades: " + " | ".join(f"{classe}: {p:.2%}" for classe, p in enumerate(probs)))
        registros.append({"indice": int(indice), "classe_correta": correto,
                          "classe_prevista": int(previsto), "acerto": acerto,
                          "probabilidades": [float(p) for p in probs]})

    if not exibir:
        return registros

    import matplotlib.pyplot as plt
    colunas = min(5, len(indices))
    linhas = math.ceil(len(indices) / colunas)
    figura, eixos = plt.subplots(linhas, colunas, figsize=(3 * colunas, 3 * linhas), squeeze=False)
    for eixo, registro in zip(eixos.flat, registros):
        eixo.imshow(x_teste[registro["indice"]].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        eixo.set_title(f"Índice {registro['indice']}\nReal: {registro['classe_correta']} | "
                       f"previsto: {registro['classe_prevista']}\n"
                       f"{'Correta' if registro['acerto'] else 'Incorreta'}")
        eixo.axis("off")
    for eixo in list(eixos.flat)[len(registros):]:
        eixo.set_visible(False)
    figura.tight_layout()
    plt.show()
    plt.close(figura)
    return registros
