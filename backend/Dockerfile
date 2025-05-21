# Dockerfile for Trading SaaS Backend (FastAPI)

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# ENV WORKERS_PER_CORE 1 # Already in .env, but can be set here too
# ENV WEB_CONCURRENCY # Calculated by Gunicorn, or set in .env

# Set work directory
WORKDIR /app

# Install system dependencies required for TA-Lib and potentially other libraries
# For TA-Lib, you might need build-essential, libta-lib-dev. The exact packages depend on the base image.
# For python:3.12-slim (Debian Bookworm), you'll need to install build-essential and wget to download and build TA-Lib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install pipenv (if you prefer to use Pipfile, otherwise skip this and use requirements.txt directly)
# RUN pip install pipenv

# Copy just the requirements file to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000 (or whatever port your FastAPI app runs on)
EXPOSE 8000

# Command to run the application
# For development with Uvicorn:
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# For production with Gunicorn (recommended):
# The Gunicorn command will use environment variables for host, port, workers, etc.
# Example: CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
# A more robust Gunicorn command might be in a shell script (e.g., entrypoint.sh) that reads from .env or environment variables.
# For now, let's stick with a simple Uvicorn command suitable for the docker-compose setup.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
