# Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

# System deps (optional but handy for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install deps (uvicorn[standard] includes watchfiles for reload)
RUN pip install -r requirements.txt

# Copy app code
COPY app ./app
# Copy exports (you can also mount them as a volume in dev)
COPY exports ./exports

# CORS origin for your Vite dev server
ENV VITE_ORIGIN=http://localhost:5173
EXPOSE 8000

# Default = prod server; dev uses docker-compose override with --reload
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
