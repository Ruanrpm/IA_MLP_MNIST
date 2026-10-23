from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
import numpy as np

from dados import (PASTA_DADOS, URL_MNIST, carregar_teste,
                   carregar_treino_validacao, hash_arquivo)
import graficos
from mlp import MLP


RAIZ = Path(__file__).resolve().parent
EXPERIMENTOS = [
    {"execucao": 1, "neuronios": [32], "taxa_aprendizado": 0.01},
    {"execucao": 2, "neuronios": [64], "taxa_aprendizado": 0.01},
    {"execucao": 3, "neuronios": [128], "taxa_aprendizado": 0.01},
    {"execucao": 4, "neuronios": [64, 32], "taxa_aprendizado": 0.01},
    {"execucao": 5, "neuronios": [64, 32], "taxa_aprendizado": 0.001},
    {"execucao": 6, "neuronios": [128, 64], "taxa_aprendizado": 0.01},
]


def salvar_json(caminho, conteudo):
    Path(caminho).write_text(json.dumps(conteudo, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")


def selecionar_melhor(resultados):
    # Desempate: menor perda de validação; depois, menor identificador.
    # Nenhum campo de teste participa da seleção.
    return max(resultados, key=lambda r: (r["acuracia_validacao_percentual"],
                                         -r["perda_validacao"], -r["execucao"]))


def matriz_confusao(y, previsoes):
    matriz = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matriz, (y, previsoes), 1)
    return matriz


def executar_experimentos(epocas=15, tamanho_lote=128, semente=42,
                          pasta_dados=PASTA_DADOS, pasta_resultados=RAIZ / "resultados",
                          configuracoes=None, threads=1):
    if epocas <= 0 or tamanho_lote <= 0 or threads <= 0:
        raise ValueError("Épocas, tamanho de lote e threads devem ser positivos.")
    configuracoes = EXPERIMENTOS if configuracoes is None else configuracoes
    if not configuracoes or len({c["execucao"] for c in configuracoes}) != len(configuracoes):
        raise ValueError("Use configurações com identificadores únicos.")
    for c in configuracoes:
        if not np.isfinite(c["taxa_aprendizado"]) or c["taxa_aprendizado"] <= 0:
            raise ValueError("A taxa de aprendizado deve ser positiva e finita.")
        MLP([784, *c["neuronios"], 10], semente)
    inicio_total = perf_counter()
    dados = carregar_treino_validacao(pasta_dados, semente)
    pasta = Path(pasta_resultados) / ("completo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    pasta.mkdir(parents=True, exist_ok=False)
    configuracao = {
        "inicio_utc": datetime.now(timezone.utc).isoformat(), "status": "treinando",
        "epocas": epocas, "tamanho_lote": tamanho_lote,
        "semente": semente, "threads": threads,
        "n_treino": len(dados.y_treino), "n_validacao": len(dados.y_validacao),
        "experimentos": configuracoes,
        "fonte_dataset": URL_MNIST, "sha256_dataset": hash_arquivo(Path(pasta_dados) / "mnist.npz"),
        "sha256_indices_treino": hashlib.sha256(dados.indices_treino.tobytes()).hexdigest(),
        "sha256_indices_validacao": hashlib.sha256(dados.indices_validacao.tobytes()).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "plataforma": platform.platform(),
        "processador": platform.processor(), "cpus_logicas": os.cpu_count(),
        "tipo_numerico": "float32", "criterio_selecao": "acuracia_validacao_final",
        "sha256_codigo": {nome: hash_arquivo(RAIZ / nome) for nome in
                          ("mlp.py", "dados.py", "experimentos.py", "graficos.py", "main.py")},
    }
    salvar_json(pasta / "configuracao.json", configuracao)
    print(f"Dados: {len(dados.y_treino)} treino / {len(dados.y_validacao)} validação. "
          "Teste ainda não carregado.", flush=True)
    print(f"Resultados: {pasta}", flush=True)
    resultados, historicos = [], {}
    for c in configuracoes:
        identificador = c["execucao"]
        print(f"\nExecução {identificador}: 784 → {c['neuronios']} → 10; "
              f"taxa = {c['taxa_aprendizado']}", flush=True)
        modelo = MLP([784, *c["neuronios"], 10], semente)
        inicial = modelo.avaliar(dados.x_treino, dados.y_treino)
        inicio = perf_counter()
        historico = modelo.treinar(
            dados.x_treino, dados.y_treino, dados.x_validacao, dados.y_validacao,
            epocas, tamanho_lote, c["taxa_aprendizado"], semente,
        )
        tempo = perf_counter() - inicio
        destino = pasta / f"execucao_{identificador}"
        destino.mkdir()
        modelo.salvar(destino / "modelo.npz")
        salvar_json(destino / "historico.json", historico)
        ultimo = historico[-1]
        resultados.append({
            "execucao": identificador, "camadas_intermediarias": len(c["neuronios"]),
            "neuronios": "-".join(map(str, c["neuronios"])),
            "taxa_aprendizado": c["taxa_aprendizado"], "epocas": epocas,
            "tamanho_lote": tamanho_lote, "semente": semente,
            "tempo_treinamento_segundos": tempo, "perda_inicial_treino": inicial["perda"],
            "perda_treino": ultimo["perda_treino"], "perda_validacao": ultimo["perda_validacao"],
            "acuracia_treino_percentual": 100 * ultimo["acuracia_treino"],
            "acuracia_validacao_percentual": 100 * ultimo["acuracia_validacao"],
        })
        historicos[identificador] = historico

    # Esta etapa acontece antes da PRIMEIRA leitura das imagens e rótulos de teste.
    melhor = selecionar_melhor(resultados)
    salvar_json(pasta / "selecao_validacao.json", {
        "melhor_execucao": melhor["execucao"],
        "acuracia_validacao_percentual": melhor["acuracia_validacao_percentual"],
        "perda_validacao": melhor["perda_validacao"], "teste_carregado": False,
        "criterio": "Maior acurácia final de validação; desempate pela menor perda e menor execução.",
    })
    configuracao["status"] = "selecao_concluida_antes_do_teste"
    salvar_json(pasta / "configuracao.json", configuracao)
    print(f"\nMelhor pela validação: execução {melhor['execucao']}. "
          "Iniciando avaliação final do teste.", flush=True)
    x_teste, y_teste = carregar_teste(pasta_dados)
    configuracao["n_teste"] = len(y_teste)
    for r in resultados:
        modelo = MLP.carregar(pasta / f"execucao_{r['execucao']}" / "modelo.npz")
        metricas = modelo.avaliar(x_teste, y_teste)
        r.update({"acertos_teste": metricas["acertos"], "erros_teste": metricas["erros"],
                  "perda_teste": metricas["perda"],
                  "acuracia_teste_percentual": 100 * metricas["acuracia"]})
        print(f"Execução {r['execucao']}: teste {metricas['acuracia']:.2%}, "
              f"{metricas['acertos']} acertos / {metricas['erros']} erros.", flush=True)
    modelo = MLP.carregar(pasta / f"execucao_{melhor['execucao']}" / "modelo.npz")
    previsoes = modelo.prever(x_teste)
    matriz = matriz_confusao(y_teste, previsoes)
    registros_classe = []
    for classe in range(10):
        total, acertos = int(matriz[classe].sum()), int(matriz[classe, classe])
        registros_classe.append({"classe": classe, "total": total, "acertos": acertos,
                                "erros": total - acertos,
                                "acuracia_percentual": 100 * acertos / total if total else None})
    salvar_json(pasta / "acertos_por_classe.json", registros_classe)
    salvar_json(pasta / "resultados.json", resultados)
    graficos.curvas(historicos, resultados, pasta)
    graficos.confusao(matriz, pasta)
    configuracao.update({"status": "concluido", "melhor_execucao": melhor["execucao"],
                         "tempo_total_segundos": perf_counter() - inicio_total,
                         "fim_utc": datetime.now(timezone.utc).isoformat()})
    salvar_json(pasta / "configuracao.json", configuracao)
    indice = Path(pasta_resultados) / "ultima_execucao_completa.json"
    salvar_json(indice, {"pasta": str(pasta.resolve()), "melhor_execucao": melhor["execucao"]})
    print(f"\nConcluído em {configuracao['tempo_total_segundos']:.2f}s. Saídas em {pasta}", flush=True)
    return pasta
