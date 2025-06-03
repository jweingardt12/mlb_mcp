# Dockerfile for Smithery-compatible MCP server
FROM python:3.11-slim

# Install build tools and libraries needed for packages like pandas and lxml
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libxml2-dev \
        libxslt1-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --log-config /app/uvicorn_log_config.json & python mcp_stdio_wrapper.py"]
