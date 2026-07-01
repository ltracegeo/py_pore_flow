FROM python:3.12-slim

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

COPY ./ ./

RUN pip install --upgrade pip setuptools wheel
RUN pip install ./src/submodules/pyflowsolver
RUN pip install .
RUN pip install petsc4py



