FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal for TA-Lib + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default: run tests
CMD ["pytest", "tests/"]
