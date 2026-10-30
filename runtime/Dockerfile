FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY pyproject.toml requirements-service.txt ./
COPY memory_bench ./memory_bench
COPY configs/text_service_hybrid.json ./configs/text_service_hybrid.json
RUN python -m pip install --no-cache-dir -c requirements-service.txt '.[service,lexical]' \
    && python -m pip freeze > /app/installed-packages.txt \
    && mkdir -p /app/var && chown -R 10001:10001 /app/var
USER 10001:10001
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=3).read()"
CMD ["python", "-m", "memory_bench.service", "--host", "0.0.0.0", "--port", "8787", "--database", "/app/var/text-hybrid-v3.sqlite3"]
