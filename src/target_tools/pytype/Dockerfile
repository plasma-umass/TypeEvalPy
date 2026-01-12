# Pull the Python base image
FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
RUN apt-get update \
    && apt-get -y install gcc

# Copy shared utilities
COPY shared_utils /tmp/shared_utils

# Copy tool-specific files
COPY pytype/requirements.txt /app/requirements.txt
COPY pytype/src /tmp/src

# Set PYTHONPATH to include shared utilities
ENV PYTHONPATH="/tmp/shared_utils:${PYTHONPATH}"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the application
CMD ["bash"]
