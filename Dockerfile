# Dockerfile for stdio-based MCP server
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Set Python optimization flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The stdio server doesn't need PORT or HTTP setup
# Smithery will handle the stdio communication

# Run the server directly
CMD ["python", "-m", "server"]