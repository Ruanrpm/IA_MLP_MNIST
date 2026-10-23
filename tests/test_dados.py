"""Verifica as partições reais e a independência do carregamento de teste."""

import unittest
from unittest.mock import patch

import numpy as np

from dados import PASTA_DADOS, carregar_teste, carregar_treino_validacao, normalizar


class TestDados(unittest.TestCase):
    @unittest.skipUnless((PASTA_DADOS / "mnist.npz").exists(), "MNIST ainda não baixado")
    def test_particoes_completas_normalizacao_e_reproducibilidade(self):
        dados = carregar_treino_validacao()
        self.assertEqual(dados.x_treino.shape, (55000, 784))
        self.assertEqual(dados.x_validacao.shape, (5000, 784))
        self.assertEqual(dados.x_treino.dtype, np.float32)
        self.assertEqual(float(dados.x_treino.min()), 0)
        self.assertEqual(float(dados.x_treino.max()), 1)
        self.assertEqual(len(np.intersect1d(dados.indices_treino, dados.indices_validacao)), 0)
        self.assertEqual(len(np.union1d(dados.indices_treino, dados.indices_validacao)), 60000)
        rapido = carregar_treino_validacao(rapido=True)
        np.testing.assert_array_equal(dados.indices_treino[:2000], rapido.indices_treino)
        np.testing.assert_array_equal(dados.indices_validacao[:500], rapido.indices_validacao)
        outra_semente = carregar_treino_validacao(semente=43, rapido=True)
        self.assertFalse(np.array_equal(rapido.indices_treino, outra_semente.indices_treino))

    def test_treino_nao_le_chaves_de_teste(self):
        class ArquivoSomenteTreino:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __getitem__(self, chave):
                if chave == "x_train":
                    return np.zeros((60000, 28, 28), dtype=np.uint8)
                if chave == "y_train":
                    return np.arange(60000, dtype=np.int64) % 10
                raise AssertionError("O treinamento tentou ler o conjunto de teste.")

        with patch("dados.baixar_mnist"), patch("dados.np.load", return_value=ArquivoSomenteTreino()):
            dados = carregar_treino_validacao(rapido=True)
            self.assertEqual(len(dados.y_treino), 2000)

    @unittest.skipUnless((PASTA_DADOS / "mnist.npz").exists(), "MNIST ainda não baixado")
    def test_teste_original_dez_classes(self):
        x, y = carregar_teste()
        self.assertEqual(x.shape, (10000, 784))
        np.testing.assert_array_equal(np.unique(y), np.arange(10))

    def test_normalizacao_rejeita_imagem_invalida(self):
        with self.assertRaises(ValueError):
            normalizar(np.zeros((2, 14, 14), dtype=np.uint8))
