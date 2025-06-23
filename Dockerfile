# Use an official, slim Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for a clean Python environment
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies that may be needed by some Python packages
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy the project packaging file and install all dependencies
# This is more efficient than copying the whole project first
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip
# This command installs your 'ragnetic' application and all its dependencies
RUN pip install .

# Copy the rest of your application's source code into the container
COPY . .

# Expose the port that the application will run on
EXPOSE 8000

# Define the command to run when the container starts.
# We use Gunicorn as a robust, multi-worker process manager for Uvicorn
# in a production-like environment.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-b", "0.0.0.0:8000"]
