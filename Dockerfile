FROM python:3.10-slim

# Set environment optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install essential system dependencies (for building wheel extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, model artifact, and entrypoint
COPY app/ ./app/
COPY model/ ./model/
COPY main.py .

# Expose FastAPI default port
EXPOSE 8000

# Execute server
CMD ["python", "main.py"]
