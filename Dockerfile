# Use the official NVIDIA CUDA image as the base
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/0.6.11/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH" UV_HTTP_TIMEOUT="600"

# Add working directory to docker
ADD . /genie3
WORKDIR /genie3

# Set up environment based on `pyproject.toml` 
RUN uv sync --frozen --extra gpu