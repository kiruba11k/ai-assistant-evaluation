FROM python:3.11-slim

LABEL maintainer="AI Assistant Evaluation Suite"
LABEL description="OSS + Frontier AI assistant evaluation framework"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root)
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY evaluation/ ./evaluation/
COPY deployment/ ./deployment/

# Create directories
RUN mkdir -p logs evaluation/results && \
    chown -R appuser:appuser /app

USER appuser

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Environment defaults (override at runtime)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "-m", "app.main"]
