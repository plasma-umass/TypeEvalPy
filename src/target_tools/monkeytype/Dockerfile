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

# Install MonkeyType
RUN pip install --upgrade pip
RUN pip install monkeytype

# Copy shared utilities
COPY shared_utils /tmp/shared_utils

# Install our runner's additional dependencies
COPY monkeytype/requirements.txt /app/runner_requirements.txt
RUN pip install -r /app/runner_requirements.txt

# Set PYTHONPATH to include shared utilities
ENV PYTHONPATH="/tmp/shared_utils:${PYTHONPATH}"

# Copy our runner code
COPY monkeytype/src /tmp/src

# Run the application
CMD ["bash"]
