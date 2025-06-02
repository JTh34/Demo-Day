FROM python:3.10-slim

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files first to optimize Docker cache
COPY pyproject.toml README.md ./

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Check critical imports
RUN python -c "from langchain_qdrant import QdrantVectorStore; print('✅ Qdrant successfully imported')"
RUN python -c "from fastapi import FastAPI; print('✅ FastAPI successfully imported')"
RUN python -c "import uvicorn; print('✅ Uvicorn successfully imported')"

# Create necessary directories with correct permissions
RUN mkdir -p /app/static && \
    mkdir -p /tmp/qdrant_storage && \
    mkdir -p /tmp/cache && \
    chmod -R 777 /tmp/qdrant_storage && \
    chmod -R 777 /tmp/cache

# Copy application files
COPY main.py .
COPY rag_system.py .
COPY agent_workflow.py .
COPY embedding_models.py .
COPY books_config.json .

# Copy the static directory with the interface
COPY static/ ./static/

# Copy the chunks file to the root
COPY all_books_preprocessed_chunks.json .

# Check for the presence of the chunks file at the root
RUN if [ -f "all_books_preprocessed_chunks.json" ]; then \
        echo "✅ Chunks file found at all_books_preprocessed_chunks.json"; \
        echo "📊 File size: $(du -h all_books_preprocessed_chunks.json)"; \
    else \
        echo "⚠️ Warning: all_books_preprocessed_chunks.json not found at root"; \
        echo "📁 Contents of root directory:"; \
        ls -la *.json || echo "No JSON files found"; \
    fi

# Ensure all files have correct permissions
RUN chmod -R 755 /app
RUN chmod +x main.py

# Expose the port for FastAPI
EXPOSE 7860

# Environment variables for FastAPI and Hugging Face Spaces
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV UVICORN_HOST="0.0.0.0"
ENV UVICORN_PORT="7860"
ENV UVICORN_LOG_LEVEL="info"

# Variables to optimize performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Variables for cache management
ENV MAX_CACHE_SIZE_MB=500
ENV MAX_CACHE_AGE_DAYS=7

# Healthcheck to verify the application is responding
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Command to launch the FastAPI application
CMD ["python", "main.py"] 