FROM python

WORKDIR /app

COPY model/ /app/model/
COPY *.py /app/
COPY LICENSE /app
COPY environment.yml /app

RUN \
    wget https://martin-krasser.de/sisr/weights-edsr-16-x4.tar.gz &&\
    wget https://martin-krasser.de/sisr/weights-wdsr-b-32-x4.tar.gz &&\
    wget https://martin-krasser.de/sisr/weights-srgan.tar.gz &&\
    tar xvfz weights-edsr-16-x4.tar.gz &&\
    tar xvfz weights-wdsr-b-32-x4.tar.gz &&\
    tar xvfz weights-srgan.tar.gz &&\
    rm -f weights-edsr-16-x4.tar.gz &&\
    rm -f weights-wdsr-b-32-x4.tar.gz &&\
    rm -f weights-srgan.tar.gz &&\
    cd /root &&\
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh &&\
    bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b &&\
    miniconda3/condabin/conda init &&\
    . /root/.bashrc &&\
    cd /app &&\
    conda env create -f environment.yml

COPY docker/*.py /app/
COPY docker/run.sh /app

ENTRYPOINT ["./run.sh"]

