#!/bin/bash

. /root/.bashrc
conda activate sisr
python ${1}.py "/working/${2}"