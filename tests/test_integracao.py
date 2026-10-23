"""Verifica o fluxo com amostras reais do MNIST em uma pasta temporária."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np

from dados import PASTA_DADOS, carregar_teste, carregar_treino_validacao
from demonstracao import demonstrar, localizar_modelo
from experimentos import executar_experimentos
from mlp import MLP


class TestIntegracao(unittest.TestCase):
    def test_localiza_modelo_apos_copiar_resultados_entre_sistemas(self):
        with tempfile.TemporaryDirectory() as temporario:
            raiz = Path(temporario)
            modelo = raiz / "completo_exemplo" / "execucao_6" / "modelo.npz"
            modelo.parent.mkdir(parents=True)
            modelo.touch()
            for origem in (r"C:\outro_pc\resultados\completo_exemplo",
                           "/app/resultados/completo_exemplo", "completo_exemplo"):
                with self.subTest(origem=origem):
                    (raiz / "ultima_execucao_completa.json").write_text(
                        json.dumps({"pasta": origem, "melhor_execucao": 6}), encoding="utf-8")
                    self.assertEqual(localizar_modelo(raiz), modelo)

    @unittest.skipUnless((PASTA_DADOS / "mnist.npz").exists(), "MNIST ainda não baixado")
    def test_fluxo_real_arquivos_selecao_e_demonstracao(self):
        with tempfile.TemporaryDirectory() as temporario, redirect_stdout(io.StringIO()):
            raiz = Path(temporario)
            chamadas_teste = []

            def carregar_teste_depois_da_selecao(*args, **kwargs):
                pastas = list(raiz.glob("completo_*"))
                self.assertEqual(len(pastas), 1)
                selecao = json.loads((pastas[0] / "selecao_validacao.json").read_text(encoding="utf-8"))
                self.assertFalse(selecao["teste_carregado"])
                self.assertEqual(len(list(pastas[0].glob("execucao_*/modelo.npz"))), 6)
                chamadas_teste.append(True)
                return carregar_teste(*args, **kwargs, rapido=True)

            dados_reduzidos = carregar_treino_validacao(rapido=True)
            with (patch("experimentos.carregar_treino_validacao", return_value=dados_reduzidos),
                  patch("experimentos.carregar_teste", side_effect=carregar_teste_depois_da_selecao)):
                pasta = executar_experimentos(epocas=1, pasta_resultados=raiz)
            self.assertEqual(len(chamadas_teste), 1)
            self.assertEqual({p.name for p in raiz.iterdir()},
                             {pasta.name, "ultima_execucao_completa.json"})
            self.assertFalse(list(raiz.rglob("*rapido*")))
            registros = json.loads((pasta / "resultados.json").read_text(encoding="utf-8"))
            self.assertEqual(len(registros), 6)
            for registro in registros:
                self.assertEqual(registro["acertos_teste"] + registro["erros_teste"], 500)
            for numero in range(1, 7):
                execucao = pasta / f"execucao_{numero}"
                self.assertEqual({p.name for p in execucao.iterdir()}, {"historico.json", "modelo.npz"})
                historico = json.loads((execucao / "historico.json").read_text(encoding="utf-8"))
                self.assertEqual(len(historico), 1)
                self.assertAlmostEqual(registros[numero - 1]["acuracia_validacao_percentual"],
                                       100 * historico[-1]["acuracia_validacao"])
            self.assertEqual({p.name for p in pasta.iterdir() if p.is_file()},
                             {"configuracao.json", "selecao_validacao.json", "resultados.json",
                              "acertos_por_classe.json", "perda.png", "acuracia.png", "matriz_confusao.png"})
            for nome in ("perda.png", "acuracia.png", "matriz_confusao.png"):
                imagem = mpimg.imread(pasta / nome)
                self.assertGreater(imagem.shape[0], 100)
                self.assertTrue(np.isfinite(imagem).all())
            x, y = carregar_teste(rapido=True)
            selecao = json.loads((pasta / "selecao_validacao.json").read_text(encoding="utf-8"))
            caminho_modelo = localizar_modelo(raiz)
            self.assertEqual(caminho_modelo, pasta / f"execucao_{selecao['melhor_execucao']}" / "modelo.npz")
            previsoes = MLP.carregar(caminho_modelo).prever(x)
            classes = json.loads((pasta / "acertos_por_classe.json").read_text(encoding="utf-8"))
            self.assertEqual(len(classes), 10)
            for registro in classes:
                classe = registro["classe"]
                total = int(np.sum(y == classe))
                acertos = int(np.sum((y == classe) & (previsoes == y)))
                self.assertEqual(registro["total"], total)
                self.assertEqual(registro["acertos"], acertos)
                self.assertEqual(registro["erros"], total - acertos)
                self.assertAlmostEqual(registro["acuracia_percentual"], 100 * acertos / total)
            self.assertEqual(sum(registro["total"] for registro in classes), 500)
            arquivos_antes = set(pasta.rglob("*"))
            demo = demonstrar(caminho_modelo, indices=[0, 1, 2], exibir=False)
            self.assertEqual(len(demo), 3)
            self.assertEqual(set(pasta.rglob("*")), arquivos_antes)
            for registro in demo:
                self.assertAlmostEqual(sum(registro["probabilidades"]), 1, places=6)
                self.assertEqual(registro["classe_prevista"], int(previsoes[registro["indice"]]))
