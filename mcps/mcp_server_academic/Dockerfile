FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY mcps/mcp_server_academic/pyproject.toml ./mcp_server_academic/

RUN pip install --no-cache-dir -e ./mcp_server_academic

COPY mcps/mcp_server_academic/ ./mcp_server_academic/

EXPOSE 8889

ENV COSCIENTIST_ACADEMIC_MCP_PORT=8889
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8889', timeout=5.0)" || exit 1

CMD ["uvicorn", "mcp_server_academic.server:app", "--host", "0.0.0.0", "--port", "8889"]
