# Pull the Python base image
FROM ghcr.io/saltudelft/type4py:latest

# Copy shared utilities
COPY shared_utils /tmp/shared_utils

# Copy tool-specific files
COPY type4py/requirements.txt /app/requirements.txt
COPY type4py/src /tmp/src

# Set PYTHONPATH to include shared utilities
ENV PYTHONPATH="/tmp/shared_utils:${PYTHONPATH}"

# Install Python dependencies
RUN pip install --upgrade pip
#RUN pip install .
RUN pip install -r /app/requirements.txt

# Start the Type4Py server and keep container running
WORKDIR /type4py/type4py/server
CMD ["bash", "run_server.sh"]