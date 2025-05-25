#!/bin/bash
set -e

printf "Setting up environment for WildGS\n"

export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
CUDA_VERSION=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
. /etc/profile.d/modules.sh
module add cuda/11.8
module add gcc/10

# Install
source /work/courses/3dv/23-2/ben/miniconda3/etc/profile.d/conda.sh
printf "Creating Conda Environment at /work/courses/3dv/23-2/envs/wildgs-$USER\n"
conda create -p /work/courses/3dv/23-2/envs/wildgs-$USER python=3.10 -y
conda activate /work/courses/3dv/23-2/envs/wildgs-$USER


conda install --channel "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install numpy==1.26.3 --no-cache-dir # do not use numpy >= v2.0.0
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.5.1+cu118.html --no-cache-dir
pip3 install -U xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

python -m pip install -e thirdparty/lietorch/ --no-cache-dir
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose/ --no-cache-dir
python -m pip install -e thirdparty/simple-knn/ --no-cache-dir

python -m pip install -e . --no-cache-dir
python -m pip install -r requirements.txt --no-cache-dir

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html --no-cache-dir

# Download weights
# mkdir -p pretrained
# curl -o pretrained/droid.pth -L "https://drive.google.com/uc?export=download&id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh"

# Clean Up
conda clean -a -y
pip cache purge

# Install datasets
bash scripts/download_bonn.sh
# bash scripts/download_tum.sh
