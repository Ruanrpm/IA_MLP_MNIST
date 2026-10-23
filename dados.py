from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import urllib.request

import numpy as np


URL_MNIST = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
PASTA_DADOS = Path(__file__).resolve().parent / "dados"


def baixar_mnist(pasta=PASTA_DADOS):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "mnist.npz"
    if not destino.exists():
        temporario = pasta / "mnist.npz.part"
        print("Baixando o MNIST (aproximadamente 11 MB)...", flush=True)
        try:
            with urllib.request.urlopen(URL_MNIST, timeout=60) as origem, temporario.open("wb") as saida:
                shutil.copyfileobj(origem, saida)
            with np.load(temporario, allow_pickle=False) as arquivo:
                if not {"x_train", "y_train", "x_test", "y_test"}.issubset(arquivo.files):
                    raise ValueError("O arquivo baixado não contém o MNIST esperado.")
            temporario.replace(destino)
        except Exception:
            temporario.unlink(missing_ok=True)
            raise
        print(f"MNIST salvo em {destino}", flush=True)
    return destino

# Impressão digital
def hash_arquivo(caminho):
    resumo = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def normalizar(imagens):
    if imagens.ndim != 3 or imagens.shape[1:] != (28, 28) or len(imagens) == 0:
        raise ValueError("São esperadas imagens de 28 × 28 pixels.")
    if imagens.dtype != np.uint8:
        raise ValueError("Os pixels originais do MNIST devem ser uint8.")
    return imagens.reshape(len(imagens), 784).astype(np.float32) / np.float32(255.0)


def _validar_originais(imagens, rotulos, quantidade):
    if imagens.shape != (quantidade, 28, 28) or rotulos.shape != (quantidade,):
        raise ValueError("As dimensões do arquivo MNIST estão incorretas.")
    if not np.issubdtype(rotulos.dtype, np.integer) or np.any((rotulos < 0) | (rotulos > 9)):
        raise ValueError("Os rótulos devem representar dígitos de 0 a 9.")


# Agrupa dados de treinamento no objeto
@dataclass
class DadosTreino:
    x_treino: np.ndarray # imagem
    y_treino: np.ndarray # rotulo / num
    x_validacao: np.ndarray
    y_validacao: np.ndarray
    indices_treino: np.ndarray
    indices_validacao: np.ndarray

# separa os conjuntos de treino e validação
def carregar_treino_validacao(pasta=PASTA_DADOS, semente=42, rapido=False):
    """Usa apenas as 60.000 imagens originais de treino, sem ler x_test/y_test."""
    with np.load(baixar_mnist(pasta), allow_pickle=False) as arquivo:
        imagens, rotulos = arquivo["x_train"], arquivo["y_train"]
    _validar_originais(imagens, rotulos, 60000)
    ordem = np.random.default_rng(semente).permutation(60000)
    indices_validacao, indices_treino = ordem[:5000], ordem[5000:]
    if rapido:
        # Subconjuntos das mesmas partições, destinados somente a desenvolvimento.
        indices_treino, indices_validacao = indices_treino[:2000], indices_validacao[:500]
    return DadosTreino(normalizar(imagens[indices_treino]), rotulos[indices_treino],
                       normalizar(imagens[indices_validacao]), rotulos[indices_validacao],  
                       indices_treino, indices_validacao)


def carregar_teste(pasta=PASTA_DADOS, rapido=False):
    """Chamado somente na avaliação final ou em uma demonstração de modelo pronto."""
    with np.load(baixar_mnist(pasta), allow_pickle=False) as arquivo:
        imagens, rotulos = arquivo["x_test"], arquivo["y_test"]
    _validar_originais(imagens, rotulos, 10000)
    if rapido:
        imagens, rotulos = imagens[:500], rotulos[:500]
    return normalizar(imagens), rotulos
