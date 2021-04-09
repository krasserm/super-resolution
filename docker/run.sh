#!/bin/bash

export PS1='$ '
. /root/.bashrc

export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

conda activate sisr
python ${1}.py "/working/${2}"
