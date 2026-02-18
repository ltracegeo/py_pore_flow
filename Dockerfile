FROM python:3.9-slim

ENV SETUPTOOLS_USE_DISTUTILS=stdlib

# Install git, C++ compiler (for pyamg), and other potential dependencies
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libfontconfig1 \
    make \
    libopenblas-dev \
    liblapack-dev \
    openmpi-bin \
    libopenmpi-dev \
    gfortran && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /py_pore_flow

# Copy only the necessary files for installation
# ./py_pore_flow/pyproject.toml ./
#COPY ./py_pore_flow/src ./
#COPY ./py_pore_flow/tests ./tests
COPY ./ ./

# Install the package
RUN pip install .
RUN pip install ./src/py_pore_flow/pyflowsolver
RUN pip install petsc4py

# Set working directory for the execution
WORKDIR /app

# No COPY here to keep the image generic for different networks
# The network files will be mounted or copied at runtime.


