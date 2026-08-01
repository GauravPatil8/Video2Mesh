FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

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
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


RUN wget -q \
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
-O miniforge.sh && \
bash miniforge.sh -b -p ${CONDA_DIR} && \
rm miniforge.sh && \
conda clean -afy

WORKDIR /workspace

RUN git clone https://github.com/GauravPatil8/Video2Mesh.git

WORKDIR /workspace/Video2Mesh

RUN mkdir -p libs && \
    git clone https://github.com/Anttwo/SuGaR.git libs/SuGaR


WORKDIR /workspace/Video2Mesh/libs/SuGaR

RUN python install.py


WORKDIR /workspace/Video2Mesh

RUN conda run -n sugar pip install -r requirements.txt

# Auto-activate environment
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate sugar" >> ~/.bashrc

CMD ["bash"]