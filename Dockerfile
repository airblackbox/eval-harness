FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Eval harness runs as CLI, invoked by integration tests
CMD ["python3", "-m", "cli.main", "--help"]
