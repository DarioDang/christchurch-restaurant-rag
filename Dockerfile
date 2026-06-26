FROM python:3.10-slim

WORKDIR /app

# System deps needed by some ML libs (numpy/scipy wheels sometimes need this)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT at runtime — same pattern as Render
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}