# The placement service. CPU PyTorch is enough: the ensemble is small and retrains in seconds.
FROM python:3.13-slim AS build
WORKDIR /src
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --prefix=/install .

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MFS_ROOT=/app
RUN useradd --create-home --uid 10001 mfs && mkdir -p /var/lib/mfs && chown mfs /var/lib/mfs
WORKDIR /app
COPY --from=build /install /usr/local
COPY config ./config
USER mfs
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
CMD ["python", "-m", "mfs.cli", "serve", "--port", "8080", "--state", "/var/lib/mfs/state.json"]
