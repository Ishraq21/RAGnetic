# sandbox/Dockerfile
FROM python:3.10-slim

# Make logs unbuffered and avoid pip cache
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# Default Matplotlib backend/config dir (runner sets these too; this keeps behavior consistent)
ENV MPLBACKEND=Agg MPLCONFIGDIR=/work/.mplconfig

# Create a non-root user to run the application
RUN useradd --create-home --shell /bin/bash sandbox-user

# Prepare sandbox dirs and a compatibility symlink for /mnt/work
# Make /mnt/work resolve to /work (handles code that uses /mnt/work)
RUN mkdir -p /work /work/.mplconfig \
 && chown -R sandbox-user:sandbox-user /work \
 && rm -rf /mnt/work && ln -s /work /mnt/work

# Set the working directory for the user
WORKDIR /home/sandbox-user

# Install core dependencies (remove duplicate seaborn)
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

# Define the entrypoint for the container (unbuffered for timely logs)
ENTRYPOINT ["python", "-u", "/usr/local/bin/runner.py"]
