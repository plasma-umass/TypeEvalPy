# Pull the Python base image
FROM python:3.12-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
#ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
RUN apt-get update \
    && apt-get -y install gcc
RUN apt-get -y install git

# Copy shared utilities
COPY shared_utils /tmp/shared_utils

# Copy tool-specific files
COPY righttyper/requirements.txt /app/requirements.txt
COPY righttyper/src /tmp/src

# Set PYTHONPATH to include shared utilities
ENV PYTHONPATH="/tmp/shared_utils:${PYTHONPATH}"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Keep the container alive
CMD ["bash"]
