# Pull the Python base image
FROM python:3.12-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set work directory
WORKDIR /app

# Install dependencies
RUN apt-get update \
    && apt-get -y install gcc python3-packaging python3-pip python3-dev python3-venv python3-setuptools git

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install -r requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python


COPY src /tmp/src

# Keep the container alive
CMD ["bash"]
