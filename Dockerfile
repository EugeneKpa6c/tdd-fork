ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Добавление репозитория Ubuntu Toolchain R с использованием прямой ссылки
RUN echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-ppa-bionic.list && \
    apt-key adv --fetch-keys https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x1E9377A2BA9EF27F

#========#
# Добавление репозитория `savoury1/gcc-defaults-9`
RUN echo "deb https://ppa.launchpadcontent.net/savoury1/gcc-defaults-9/ubuntu bionic main" > /etc/apt/sources.list.d/savoury1-gcc-defaults-9.list && \
    echo "deb-src https://ppa.launchpadcontent.net/savoury1/gcc-defaults-9/ubuntu bionic main" >> /etc/apt/sources.list.d/savoury1-gcc-defaults-9.list

# Импорт ключа GPG репозитория `savoury1/gcc-defaults-9`
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E996735927E427A733BB653E374C7797FB006459

# Обновление списка пакетов после добавления нового репозитория
RUN apt-get update
#========#

WORKDIR /app

# Устанавливаем необходимые пакеты и библиотеки
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN echo "Downloading Miniconda with wget..." \
 && wget --no-check-certificate -O /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && echo "Setting execute permission for Miniconda installer..." \
 && chmod +x /miniconda.sh \
 && echo "Installing Miniconda..." \
 && /miniconda.sh -b -p /miniconda \
 && echo "Removing Miniconda installer..." \
 && rm /miniconda.sh

# Отключение SSL-проверки для Conda
RUN /miniconda/bin/conda config --set ssl_verify false

# Create a Python 3.8 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py38 python=3.8 \
 && /miniconda/bin/conda clean -ya

# сокращаем установку
ENV DEBIAN_FRONTEND=noninteractive

ENV CONDA_DEFAULT_ENV=py38
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV

ENV PATH=/usr/local/cuda/bin:$CONDA_PREFIX/bin:/miniconda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# добавляем все катологи для скачивания
COPY pip.conf pip.conf
ENV PIP_CONFIG_FILE pip.conf

RUN apt-get update && apt-get install -y git

RUN apt update
RUN apt install curl -y

RUN pip3 install --upgrade pip

#RUN pip3 install --trusted-host pypi.org --trusted-host download.openmmlab.com --trusted-host pip.pypa.io --trusted-host files.pythonhosted.org -r requirements.txt
#RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN conda install -y pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Клонируем репозиторий VisualizerX
RUN git clone https://github.com/chenzhik/VisualizerX /VisualizerX

# Переходим в директорию VisualizerX
WORKDIR /VisualizerX

# Устанавливаем зависимости VisualizerX
RUN pip3 install --trusted-host pypi.org --trusted-host pip.pypa.io --trusted-host files.pythonhosted.org bytecode scikit-learn

# Устанавливаем сам VisualizerX
RUN python setup.py install

# Возвращаем рабочий каталог обратно на /app для последующих инструкций
WORKDIR /app


RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.3'

RUN pip3 install --trusted-host pypi.org --trusted-host pip.pypa.io --trusted-host files.pythonhosted.org opencv-python

COPY . /app

#COPY edit_detectron2.sh /edit_detectron2.sh
RUN chmod +x /app/edit_detectron2.sh && /app/edit_detectron2.sh

# Устанавливаем рабочий каталог
WORKDIR /app

# Запускаем оболочку bash при старте контейнера
CMD ["bash"]