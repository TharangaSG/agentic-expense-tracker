# Use a slim Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first
COPY pyproject.toml ./

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# Copy project source code
COPY src/ ./src/

# Expose port
EXPOSE 8001

# Run the application 
CMD ["python", "-m", "src.interfaces.whatsapp.whatsapp_app"]
