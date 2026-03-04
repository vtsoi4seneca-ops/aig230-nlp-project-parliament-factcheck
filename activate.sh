#!/bin/bash
# activate.sh - Source this to activate uv environment

cd ~/parliament-factcheck
source .venv/bin/activate

# Set DGX Spark specific environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "✅ Parliament Fact-Check environment activated"
echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

