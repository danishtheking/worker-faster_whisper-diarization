# CUDA 12.1 — compatible with driver >= 530.30 (maximum RunPod node coverage)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Set working directory
WORKDIR /

# Update and upgrade the system packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential python3-pip python3-venv -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.6 with CUDA 12.1 support
RUN pip install torch==2.6.0+cu121 torchaudio==2.6.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install huggingface_hub[hf_xet] && \
    pip install -r /requirements.txt --no-cache-dir

# Copy and run script to fetch models
COPY builder/fetch_models.py /fetch_models.py
RUN python3 /fetch_models.py && \
    rm /fetch_models.py

# Copy handler, models and other code
COPY src .
COPY models models/

# test input that will be used when the container runs outside of runpod
COPY test_input.json .

# Set default command
CMD python3 -u /rp_handler.py
