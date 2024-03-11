# Decimas

## Installation Instructions

**Note:** This package was tested and run on Ubuntu 22.04 using two Nvidia 4090s.

```bash
conda create -n decimas python=3.10
conda activate decimas
pip install "torch==2.1.2" tensorboard
pip install  --upgrade   "transformers==4.36.2"   "datasets==2.16.1"   "accelerate==0.26.1"   "evaluate==0.4.1"   "bitsandbytes==0.42.0"
pip install git+https://github.com/huggingface/trl@a3c5b7178ac4f65569975efadc97db2f3749c65e --upgrade
pip install git+https://github.com/huggingface/peft@4a1559582281fc3c9283892caea8ccef1d6f5a4f --upgrade
pip install ninja packaging
conda install nvidia::cuda-nvcc
MAX_JOBS=4 pip install flash-attn --no-build-isolation

```