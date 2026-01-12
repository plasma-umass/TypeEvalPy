# Pull the Python base image
FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get -y install git gcc \
    && rm -rf /var/lib/apt/lists/*

# Clone QuAC repository
RUN git clone https://github.com/jifengwu2k/quac.git /app/quac

# Install QuAC's dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/quac/requirements.txt

# Copy shared utilities
COPY shared_utils /tmp/shared_utils

# Install our runner's additional dependencies
COPY quac/requirements.txt /app/runner_requirements.txt
RUN pip install -r /app/runner_requirements.txt

# Add QuAC and shared_utils to Python path
ENV PYTHONPATH="/tmp/shared_utils:/app/quac:${PYTHONPATH}"

# Copy our runner code
COPY quac/src /tmp/src

# Run the application
CMD ["bash"]
