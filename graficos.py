import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Salva figuras mesmo em computadores sem interface gráfica.
import matplotlib.pyplot as plt


def curvas(historicos, resultados, pasta):
    colunas = min(3, len(resultados))
    linhas = math.ceil(len(resultados) / colunas)
    for metrica, titulo, arquivo in (
        ("perda", "Entropia cruzada", "perda.png"),
        ("acuracia", "Acurácia (%)", "acuracia.png"),
    ):
        figura, eixos = plt.subplots(linhas, colunas, figsize=(5 * colunas, 3.5 * linhas),
                                    squeeze=False)
        for eixo, resultado in zip(eixos.flat, resultados):
            historico = historicos[resultado["execucao"]]
            epocas = [h["epoca"] for h in historico]
            fator = 100 if metrica == "acuracia" else 1
            for conjunto, nome in (("treino", "Treinamento"), ("validacao", "Validação")):
                eixo.plot(epocas, [h[f"{metrica}_{conjunto}"] * fator for h in historico],
                          label=nome)
            eixo.set_title(f"Execução {resultado['execucao']}: {resultado['neuronios']}\n"
                           f"Taxa = {resultado['taxa_aprendizado']}")
            eixo.set_xlabel("Época")
            eixo.set_ylabel(titulo)
            eixo.grid(alpha=0.25)
            eixo.legend()
        for eixo in list(eixos.flat)[len(resultados):]:
            eixo.set_visible(False)
        figura.tight_layout()
        figura.savefig(Path(pasta) / arquivo, dpi=180)
        plt.close(figura)


def confusao(matriz, pasta):
    figura, eixo = plt.subplots(figsize=(8, 7))
    imagem = eixo.imshow(matriz, cmap="Blues")
    for verdadeiro in range(10):
        for previsto in range(10):
            eixo.text(previsto, verdadeiro, str(matriz[verdadeiro, previsto]),
                      ha="center", va="center", fontsize=8,
                      color="white" if matriz[verdadeiro, previsto] > matriz.max() / 2 else "black")
    eixo.set_xticks(range(10))
    eixo.set_yticks(range(10))
    eixo.set_xlabel("Dígito previsto")
    eixo.set_ylabel("Dígito verdadeiro")
    eixo.set_title("Matriz de confusão — modelo selecionado pela validação")
    figura.colorbar(imagem, ax=eixo, label="Quantidade de imagens")
    figura.tight_layout()
    figura.savefig(Path(pasta) / "matriz_confusao.png", dpi=180)
    plt.close(figura)
