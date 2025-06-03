# Dockerfile for Smithery-compatible MCP server
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --log-config /app/uvicorn_log_config.json & python mcp_stdio_wrapper.py"]
