import argparse
import json
import os
from pathlib import Path
import sys
import unittest


RAIZ = Path(__file__).resolve().parent


def argumentos():
    parser = argparse.ArgumentParser(description="Reconhecimento de dígitos com MLP e Backpropagation manual.")
    comandos = parser.add_subparsers(dest="comando", required=True)
    for nome, descricao in (("treinar", "Treinar uma arquitetura configurável."),
                            ("experimentos", "Executar as seis configurações do trabalho.")):
        comando = comandos.add_parser(nome, help=descricao)
        comando.add_argument("--epocas", type=int, default=15, help="Quantidade de épocas (padrão: 15).")
        comando.add_argument("--lote", type=int, default=128, help="Imagens por mini-batch (padrão: 128).")
        comando.add_argument("--semente", type=int, default=42)
        comando.add_argument("--threads", type=int, default=1, help="Threads das operações matriciais (padrão: 1).")
        comando.add_argument("--pasta-dados", type=Path, default=RAIZ / "dados")
        comando.add_argument("--pasta-resultados", type=Path, default=RAIZ / "resultados")
        if nome == "treinar":
            comando.add_argument("--camadas", type=int, nargs="+", default=[128], help="Neurônios intermediários, ex.: 64 32.")
            comando.add_argument("--taxa", type=float, default=0.01)
        else:
            comando.add_argument("--configuracoes", type=Path, help="JSON com lista de configurações; ver configuracoes.json.")
    demo = comandos.add_parser("demonstrar", help="Reconhecer imagens de teste usando um modelo salvo.")
    demo.add_argument("--modelo", type=Path, help="Padrão: melhor modelo da última execução completa.")
    escolha = demo.add_mutually_exclusive_group()
    escolha.add_argument("--indices", type=int, nargs="+", help="Índices de teste de 0 a 9.999.")
    escolha.add_argument("--quantidade", type=int, help="Quantidade de imagens aleatórias distintas.")
    demo.add_argument("--semente", type=int, default=42)
    demo.add_argument("--pasta-dados", type=Path, default=RAIZ / "dados")
    demo.add_argument("--pasta-resultados", type=Path, default=RAIZ / "resultados")
    demo.add_argument("--sem-exibir", action="store_true", help="Mostra as previsões apenas no terminal.")
    comandos.add_parser("verificar-gradientes", help="Verificar todos os gradientes de uma rede pequena.")
    comandos.add_parser("testar", help="Executar a suíte de testes com unittest.")
    return parser, parser.parse_args()


def main():
    # O terminal do Windows pode usar uma codificação que não aceita símbolos.
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")
    parser, args = argumentos()
    # Definir ANTES de importar NumPy evita excesso de threads em matrizes pequenas.
    threads = getattr(args, "threads", 1)
    if threads <= 0:
        parser.error("--threads deve ser positivo.")
    for nome in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[nome] = str(threads)
    if args.comando == "testar":
        testes = unittest.defaultTestLoader.discover(str(RAIZ / "tests"))
        resultado = unittest.TextTestRunner(verbosity=2).run(testes)
        return 0 if resultado.wasSuccessful() else 1
    try:
        if args.comando == "verificar-gradientes":
            from mlp import verificar_gradientes
            print(json.dumps(verificar_gradientes(), ensure_ascii=False, indent=2))
        elif args.comando in ("treinar", "experimentos"):
            from experimentos import executar_experimentos
            configuracoes = None
            if args.comando == "treinar":
                configuracoes = [{"execucao": 1, "neuronios": args.camadas, "taxa_aprendizado": args.taxa}]
            elif args.configuracoes:
                configuracoes = json.loads(args.configuracoes.read_text(encoding="utf-8-sig"))
            executar_experimentos(args.epocas, args.lote, args.semente,
                                  args.pasta_dados, args.pasta_resultados, configuracoes, threads)
        elif args.comando == "demonstrar":
            from demonstracao import demonstrar, localizar_modelo
            caminho = args.modelo or localizar_modelo(args.pasta_resultados)
            demonstrar(caminho, args.indices, args.quantidade, args.semente,
                       args.pasta_dados, exibir=not args.sem_exibir)
    except (ValueError, OSError, FloatingPointError, KeyError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
