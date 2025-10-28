# RAG Application Docker Setup Guide

This guide explains how to build and run the RAG (Retrieval-Augmented Generation) application using Docker and Docker Compose.

## Prerequisites

- Docker installed on your system
- Docker Compose installed (usually comes with Docker Desktop)
- At least 4GB of available RAM (for ML models)
- OpenAI API key (for the application to work)

## Quick Start

1. **Clone or navigate to the project directory**
   ```bash
   cd rag_app
   ```

2. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit the .env file and add your OpenAI API key
   # Replace 'your_openai_api_key_here' with your actual API key
   ```

3. **Build and start the application**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - API: http://localhost:8000
   - Interactive API docs: http://localhost:8000/docs

5. **Stop the application**
   ```bash
   docker-compose down
   ```

## Detailed Docker Setup

### Building the Docker Image

You can build the Docker image in two ways:

#### Option 1: Using Docker Compose (Recommended)
```bash
docker-compose build
```

#### Option 2: Using Docker directly
```bash
docker build -t rag-app .
```

### Running the Container

#### Option 1: Using Docker Compose (Recommended)
```bash
# Start in background
docker-compose up -d

# Start with logs visible
docker-compose up

# View logs
docker-compose logs -f rag-app
```

#### Option 2: Using Docker directly
```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key_here \
  -v $(pwd)/research_db:/app/research_db \
  -v $(pwd)/ocr_data_output:/app/ocr_data_output \
  --name rag-app \
  rag-app
```

## File Structure Explanation

### Dockerfile Breakdown

Let's go through the Dockerfile line by line:

```dockerfile
# Multi-stage build for optimized production image
FROM python:3.11-slim as base
```
- **Purpose**: Sets the base image for our container
- **What it does**: Uses Python 3.11 slim version (smaller size, fewer packages)
- **Why slim**: Reduces image size while keeping essential Python functionality

```dockerfile
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
```
- **Purpose**: Installs system-level dependencies needed for Python packages
- **gcc/g++**: Compilers needed to build some Python packages (like numpy, scipy)
- **rm -rf /var/lib/apt/lists/***: Cleans up package cache to reduce image size

```dockerfile
# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser
```
- **Purpose**: Creates a non-root user for security best practices
- **groupadd -r appuser**: Creates a system group named 'appuser'
- **useradd -r -g appuser -m appuser**: Creates a system user in the appuser group with a home directory
- **Why**: Running containers as root is a security risk

```dockerfile
# Set working directory
WORKDIR /app
```
- **Purpose**: Sets the default directory for subsequent commands
- **What it does**: All following commands will run from /app directory
- **Creates**: The directory if it doesn't exist

```dockerfile
# Install Python dependencies
COPY requirements.txt* ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
```
- **COPY requirements.txt* ./**: Copies requirements.txt file to container
- **pip install --upgrade pip**: Updates pip to latest version
- **--no-cache-dir**: Doesn't store pip cache, reducing image size
- **Why separate**: Copying requirements first allows Docker to cache this layer

```dockerfile
# Copy application code
COPY --chown=appuser:appuser *.py ./
```
- **Purpose**: Copies Python application files to container
- **--chown=appuser:appuser**: Sets ownership to our non-root user
- **\*.py**: Copies all Python files in current directory

```dockerfile
# Create directories for data with proper permissions
RUN mkdir -p /app/research_db /app/ocr_data_output /app/.cache/huggingface /app/.cache/sentence_transformers /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser
```
- **mkdir -p**: Creates directories (and parent directories if needed)
- **research_db**: For ChromaDB database storage
- **ocr_data_output**: For OCR processing results
- **.cache directories**: For ML model caching
- **chown -R**: Recursively changes ownership to appuser

```dockerfile
# Switch to non-root user
USER appuser
```
- **Purpose**: Switches from root to appuser for security
- **Effect**: All subsequent commands run as appuser

```dockerfile
# Set environment variables for HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
```
- **Purpose**: Configures where ML models are cached
- **HF_HOME**: HuggingFace cache directory
- **TRANSFORMERS_CACHE**: Transformers library cache
- **SENTENCE_TRANSFORMERS_HOME**: Sentence transformers cache

```dockerfile
# Expose port
EXPOSE 8000
```
- **Purpose**: Documents which port the application uses
- **Note**: Doesn't actually publish the port (done with -p flag or docker-compose)

```dockerfile
# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1
```
- **Purpose**: Defines how Docker checks if container is healthy
- **--interval=30s**: Check every 30 seconds
- **--timeout=30s**: Wait max 30 seconds for response
- **--start-period=5s**: Wait 5 seconds before first check
- **--retries=3**: Try 3 times before marking unhealthy

```dockerfile
# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
- **Purpose**: Default command when container starts
- **uvicorn**: ASGI server for FastAPI
- **app:app**: Module and application instance
- **--host 0.0.0.0**: Listen on all interfaces
- **--port 8000**: Listen on port 8000

### docker-compose.yml Breakdown

```yaml
version: '3.8'
```
- **Purpose**: Specifies Docker Compose file format version
- **3.8**: Modern version with good feature support

```yaml
services:
  rag-app:
```
- **Purpose**: Defines services (containers) in the application
- **rag-app**: Name of our service

```yaml
    build: .
```
- **Purpose**: Tells Docker Compose to build image from current directory
- **Alternative**: Could use `image: rag-app` if image already exists

```yaml
    ports:
      - "8000:8000"
```
- **Purpose**: Maps host port to container port
- **Format**: "host_port:container_port"
- **Effect**: Makes app accessible at localhost:8000

```yaml
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TOKENIZERS_PARALLELISM=false
```
- **Purpose**: Sets environment variables in container
- **${OPENAI_API_KEY}**: Reads from .env file or system environment
- **TOKENIZERS_PARALLELISM=false**: Prevents tokenizer threading issues

```yaml
    volumes:
      - ./research_db:/app/research_db
      - ./ocr_data_output:/app/ocr_data_output
      - huggingface_cache:/app/.cache/huggingface
      - sentence_transformers_cache:/app/.cache/sentence_transformers
```
- **Purpose**: Mounts directories between host and container
- **./research_db:/app/research_db**: Bind mount for database persistence
- **./ocr_data_output:/app/ocr_data_output**: Bind mount for OCR outputs
- **Named volumes**: For ML model caching (managed by Docker)

```yaml
    restart: unless-stopped
```
- **Purpose**: Container restart policy
- **unless-stopped**: Restart automatically unless manually stopped

```yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```
- **Purpose**: Overrides Dockerfile healthcheck with custom settings
- **test**: Command to run for health check
- **start_period: 40s**: Longer wait for ML models to load

```yaml
volumes:
  huggingface_cache:
  sentence_transformers_cache:
```
- **Purpose**: Defines named volumes for persistent storage
- **Managed by Docker**: Docker handles storage location
- **Persistent**: Data survives container recreation

## Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Change port in docker-compose.yml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

2. **Permission denied errors**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

3. **Out of memory errors**
   ```bash
   # Increase Docker memory limit in Docker Desktop settings
   # Minimum 4GB recommended
   ```

4. **Container won't start**
   ```bash
   # Check logs
   docker-compose logs rag-app
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

### Useful Commands

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f rag-app

# Execute commands in running container
docker-compose exec rag-app bash

# Restart service
docker-compose restart rag-app

# Remove everything (containers, networks, volumes)
docker-compose down -v

# Update and restart
docker-compose pull && docker-compose up -d
```

## API Usage

Once running, you can:

1. **Check health**: GET http://localhost:8000/
2. **View API docs**: http://localhost:8000/docs
3. **Ask questions**: POST http://localhost:8000/ask

Example API call:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

## Performance Notes

- **First startup**: Takes longer due to ML model downloads
- **Subsequent starts**: Faster due to cached models
- **Memory usage**: ~2-4GB depending on models loaded
- **Storage**: Models cache ~1-2GB on disk

## Security Considerations

- Application runs as non-root user
- No sensitive data in Docker image
- Environment variables for secrets
- Health checks for monitoring
- Minimal base image for reduced attack surface