# syntax = docker/dockerfile:experimental
FROM ubuntu:20.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

# Install base packages (like Python and pip)
RUN apt update && apt install -y curl zip git lsb-release software-properties-common apt-transport-https vim wget python3 python3-pip libgl1-mesa-glx jq

# Update pip
RUN python3 -m pip install --upgrade pip

# Clone EdgeLab
RUN git clone https://github.com/Seeed-Studio/EdgeLab

# Install EdgeLab
RUN cd EdgeLab && \
    ./scripts/setup_docker.sh

WORKDIR /scripts

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]