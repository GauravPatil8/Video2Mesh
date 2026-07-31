FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN wget -q \
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
-O miniforge.sh && \
bash miniforge.sh -b -p ${CONDA_DIR} && \
rm miniforge.sh && \
conda clean -afy


WORKDIR /workspace

COPY . /workspace/implementation
WORKDIR /workspace/implementation

RUN mkdir -p libs && git clone https://github.com/Anttwo/SuGaR.git libs/SuGaR

WORKDIR /workspace/implementation/libs/SuGaR

RUN python install.py

CMD ["bash"]