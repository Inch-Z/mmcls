ARG PYTORCH="1.8.1"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"
#ubun 18.04
RUN cat /etc/issue
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN rm /etc/apt/sources.list
RUN echo deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse >> /etc/apt/sources.list
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install openmim
RUN mim install mmcv-full

# Install MMClassification
RUN conda clean --all
RUN git clone https://github.com/Inch-Z/mmcls.git
WORKDIR ./mmcls
RUN pwd
RUN pip install --no-cache-dir -e .
