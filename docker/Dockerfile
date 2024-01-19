# Modified from the TensorFlow 2.3 Dockerfile:
# https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile

# Use the Nvidia CUDA image as a base.
FROM nvcr.io/nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 as base

# Author and Maintainer 
MAINTAINER Kimmo Kartasalo "kimmo.kartasalo@ki.se"

# Avoid interactive prompts.
ENV DEBIAN_FRONTEND noninteractive

# Switch to the user "docker".
RUN useradd -ms /bin/bash docker
RUN su docker

# Create workdir.
RUN mkdir app
WORKDIR app/
COPY . .

# Install basic utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        bzip2 \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*    

# Update Nvidia repo keys automatically.
RUN rm /etc/apt/sources.list.d/cuda.list
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-command-line-tools-10-1 \
        libcublas10=10.2.1.243-1 \ 
        cuda-nvrtc-10-1 \
        cuda-cufft-10-1 \
        cuda-curand-10-1 \
        cuda-cusolver-10-1 \
        cuda-cusparse-10-1 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip && \
    rm -rf /var/lib/apt/lists/*

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install python and pip.
RUN apt-get update && apt-get install -y python3.6 python3-pip

# Update pip and setuptools.
RUN python3 -m pip --no-cache-dir install --upgrade pip setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Install python packages.
RUN python3 -m pip install --no-cache-dir -r requirements.txt
