FROM ubuntu:20.04

WORKDIR /app

COPY model/ /app/model/
COPY *.py /app/
COPY LICENSE /app
COPY environment.yml /app

RUN \
    apt-get update &&\
    DEBIAN_FRONTEND='noninteractive' apt-get install -y python wget nvidia-cuda-dev=10.1.243-3 &&\
    wget https://martin-krasser.de/sisr/weights-edsr-16-x4.tar.gz &&\
    wget https://martin-krasser.de/sisr/weights-wdsr-b-32-x4.tar.gz &&\
    wget https://martin-krasser.de/sisr/weights-srgan.tar.gz &&\
    tar xvfz weights-edsr-16-x4.tar.gz &&\
    tar xvfz weights-wdsr-b-32-x4.tar.gz &&\
    tar xvfz weights-srgan.tar.gz &&\
    rm -f weights-edsr-16-x4.tar.gz &&\
    rm -f weights-wdsr-b-32-x4.tar.gz &&\
    rm -f weights-srgan.tar.gz &&\
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb &&\
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb &&\
    apt-get install -y ./libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb &&\
    apt-get install -y ./libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb &&\
    rm -f ./libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb &&\
    rm -f ./libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb &&\
    cd /root &&\
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh &&\
    bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b &&\
    rm -f Miniconda3-py39_4.9.2-Linux-x86_64.sh &&\
    miniconda3/condabin/conda init bash &&\
    bash -c " \
        export PS1='$ ' &&\
        . /root/.bashrc &&\
        cd /app &&\
        conda env create -f environment.yml \
    " &&\
    apt-get clean

COPY docker/*.py /app/
COPY docker/run.sh /app

ENTRYPOINT ["./run.sh"]

