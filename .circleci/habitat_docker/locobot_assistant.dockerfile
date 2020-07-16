# syntax = docker/dockerfile:experimental
# FROM nvidia/cudagl:10.1-base-ubuntu18.04 as base
FROM soumith/pyrobot-and-habitat:latest as base

RUN apt-get update && apt-get install -y \
    cmake \
	curl \
    g++ \
    clang-format \
    git \
    htop \
    libboost-all-dev \
    libeigen3-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    make \
    python3-dev \
    python3-pip \
    zlib1g-dev 

COPY mcdeploy.key /mcdeploy.key

RUN mkdir -p /root/.ssh
RUN cp mcdeploy.key /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
SHELL ["/bin/bash", "-c"]

# Environment 
ENV GIT_SSH_COMMAND "ssh -i /mcdeploy.key -o StrictHostKeyChecking=no"

# RUN git clone git@github.com:fairinternal/minecraft.git

RUN conda create -n minecraft_env python==3.7.4 pip numpy scikit-learn==0.19.1 pytorch==1.4.0 torchvision -c conda-forge -c pytorch && \
    # cd minecraft && \
    conda init bash && \
    source ~/.bashrc && \
    source activate /root/miniconda3/envs/minecraft_env && \
    curl https://raw.githubusercontent.com/fairinternal/minecraft/master/requirements.txt?token=ANXAMPCPGDKWBRT4VI2C7E27B6F54 -o requirements.txt && \
    pip install -r requirements.txt && \
    conda install -c conda-forge opencv && \
    pip install webcolors==1.10 && \
    pip install dlib==19.19.0 && \
    pip install face_recognition && \
    pip install facenet-pytorch && \
    pip install filelock && \
    conda install -c conda-forge quaternion && \
    pip install cython && \
    pip install Pyro4 && \
    pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' && \
    python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.4/index.html