FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ENV AML_RETRIEVER=lexical AML_ALLOW_EMBEDDING_API=0 AML_CODE_IDENTIFIERS=1 AML_CACHE_USERS=2
ENV AML_TEXT_DATABASE=/app/var/multitrack-lexical-v1.sqlite3
WORKDIR /app
COPY pyproject.toml requirements-service.txt ./
COPY memory_bench ./memory_bench
COPY scripts/smoke_multitrack_service.py ./scripts/
RUN python -m pip install --no-cache-dir -c requirements-service.txt '.[service,lexical]' \
    && python -m pip freeze > /app/installed-packages.txt \
    && mkdir -p /app/var && chown -R 10001:10001 /app/var
USER 10001:10001
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=3).read()"
CMD ["python", "-m", "memory_bench.service.multitrack", "--host", "0.0.0.0", "--port", "8787"]
