"""Verificações matemáticas independentes e comportamento do treinamento."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlp import MLP, entropia_cruzada, one_hot, softmax, verificar_gradientes


class TestMLP(unittest.TestCase):
    def test_gradientes_por_diferencas_finitas(self):
        resultado = verificar_gradientes()
        self.assertTrue(resultado["aprovado"])
        self.assertEqual(resultado["parametros_verificados"], 64)
        self.assertLess(resultado["maior_erro_absoluto"], 1e-7)

    def test_softmax_e_perda_extremas(self):
        logits = np.array([[1000., -1000., 0.], [-1000., -1000., -1000.]])
        probabilidades = softmax(logits)
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1)
        self.assertTrue(np.isfinite(probabilidades).all())
        perda = entropia_cruzada(logits, one_hot(np.array([1, 0]), 3, np.float64))
        self.assertAlmostEqual(perda, (2000 + np.log(3)) / 2)

    def test_dimensoes_forward_e_gradientes(self):
        modelo = MLP([784, 64, 32, 10])
        x = np.random.default_rng(1).random((7, 784), dtype=np.float32)
        probabilidades = modelo.forward(x)
        self.assertEqual(probabilidades.shape, (7, 10))
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1, atol=1e-6)
        self.assertTrue(np.all((probabilidades >= 0) & (probabilidades <= 1)))
        _, pesos, bias = modelo.gradientes(x, np.arange(7))
        self.assertEqual([p.shape for p in pesos], [(784, 64), (64, 32), (32, 10)])
        self.assertEqual([b.shape for b in bias], [(1, 64), (1, 32), (1, 10)])

    def test_perda_diminui_e_preve_dez_classes(self):
        # Dez padrões independentes; um teste de aprendizado, não uma métrica MNIST.
        x = np.tile(np.eye(10, dtype=np.float32), (12, 1))
        y = np.tile(np.arange(10), 12)
        modelo = MLP([10, 24, 10], semente=3)
        inicial = modelo.calcular_perda(x, y)
        modelo.treinar(x, y, x, y, epocas=60, tamanho_lote=30,
                       taxa_aprendizado=0.3, semente=3, verbose=False)
        self.assertLess(modelo.calcular_perda(x, y), inicial / 5)
        np.testing.assert_array_equal(modelo.prever(np.eye(10, dtype=np.float32)), np.arange(10))

    def test_treinamento_reproduzivel(self):
        gerador = np.random.default_rng(11)
        x, y = gerador.random((31, 4), dtype=np.float32), gerador.integers(0, 3, 31)
        modelos = [MLP([4, 7, 3], semente=11) for _ in range(2)]
        historicos = [m.treinar(x, y, x[:9], y[:9], epocas=3, tamanho_lote=8,
                                semente=11, verbose=False) for m in modelos]
        for a, b in zip(modelos[0].pesos + modelos[0].bias, modelos[1].pesos + modelos[1].bias):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(historicos[0], historicos[1]):
            for campo in ("perda_treino", "perda_validacao", "acuracia_treino", "acuracia_validacao"):
                self.assertEqual(a[campo], b[campo])

    def test_salvar_carregar_preserva_probabilidades(self):
        modelo = MLP([4, 5, 3])
        x = np.ones((3, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporario:
            caminho = Path(temporario) / "modelo.npz"
            modelo.salvar(caminho)
            carregado = MLP.carregar(caminho)
            np.testing.assert_array_equal(modelo.forward(x), carregado.forward(x))

    def test_entradas_invalidas(self):
        with self.assertRaises(ValueError):
            MLP([784, 0, 10])
        with self.assertRaises(ValueError):
            one_hot(np.array([10]), 10)
        with self.assertRaises(ValueError):
            MLP([4, 5, 3]).forward(np.ones((2, 5)))
        with self.assertRaises(ValueError):
            MLP([4, 5, 3]).forward(np.full((2, 4), np.nan))
