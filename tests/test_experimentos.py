"""Verifica a seleção exclusivamente pela validação e a contagem de classes."""

import unittest

import numpy as np

from experimentos import EXPERIMENTOS, matriz_confusao, selecionar_melhor


class TestExperimentos(unittest.TestCase):
    def test_selecao_ignora_resultado_teste(self):
        candidatos = [
            {"execucao": 1, "acuracia_validacao_percentual": 92, "perda_validacao": 0.2,
             "acuracia_teste_percentual": 100},
            {"execucao": 2, "acuracia_validacao_percentual": 94, "perda_validacao": 0.25,
             "acuracia_teste_percentual": 70},
        ]
        self.assertEqual(selecionar_melhor(candidatos)["execucao"], 2)
        candidatos[0]["acuracia_validacao_percentual"] = 94
        self.assertEqual(selecionar_melhor(candidatos)["execucao"], 1)

    def test_confusao_conta_acertos_e_erros(self):
        y, p = np.array([0, 1, 1, 9]), np.array([0, 1, 2, 0])
        matriz = matriz_confusao(y, p)
        self.assertEqual(matriz.shape, (10, 10))
        self.assertEqual(int(matriz.sum()), 4)
        self.assertEqual(int(np.trace(matriz)), 2)
        self.assertEqual(int(matriz[1, 2]), 1)

    def test_seis_configuracoes_do_enunciado(self):
        self.assertEqual([c["neuronios"] for c in EXPERIMENTOS],
                         [[32], [64], [128], [64, 32], [64, 32], [128, 64]])
        self.assertEqual([c["taxa_aprendizado"] for c in EXPERIMENTOS],
                         [0.01, 0.01, 0.01, 0.01, 0.001, 0.01])
