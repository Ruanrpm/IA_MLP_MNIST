FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt \
    && python -m pip check

# O .dockerignore inclui somente codigo, testes, configuracoes e o MNIST local.
COPY . .
# Se o dataset nao veio na copia, baixa durante o build para permitir uso offline.
RUN mkdir -p resultados \
    && python -c "from dados import baixar_mnist; baixar_mnist()"

ENTRYPOINT ["python", "main.py"]
CMD ["experimentos", "--configuracoes", "configuracoes.json"]
