# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*
    
# Copy dependency file first (better layer caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    httpx>=0.25.0 \
    google-genai>=1.28.0 \
    openai>=1.99.1 \
    pydantic-settings>=2.10.1 \
    pydantic>=2.0.0

# Copy project source code
COPY src/ ./src/
COPY src/interfaces/whatsapp/whatsapp_app.py ./

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "whatsapp_app.py"]
