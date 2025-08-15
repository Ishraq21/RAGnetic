# sandbox/Dockerfile
# Use a minimal Python image as a base
FROM python:3.10-slim

# Create a non-root user to run the application
RUN useradd --create-home --shell /bin/bash sandbox-user

# Set the working directory for the user
WORKDIR /home/sandbox-user

# Install core dependencies. Add others as needed.
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn requests pyarrow fastapi uvicorn matplotlib seaborn jsonschema papermill textblob

# Preload NLTK corpora required by TextBlob
RUN python -m textblob.download_corpora

# Copy the runner script into the image
COPY runner.py /usr/local/bin/runner.py

# Ensure the runner script is executable
RUN chmod +x /usr/local/bin/runner.py

# Switch to the non-root user
USER sandbox-user

# Define the entrypoint for the container
ENTRYPOINT ["python", "/usr/local/bin/runner.py"]