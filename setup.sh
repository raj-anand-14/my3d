#!/usr/bin/env bash
# One-time setup for the my3d starter (Linux / WSL2 with an NVIDIA GPU, 8GB+ VRAM).
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu121}"   # match your CUDA: cu118, cu121, cu124 ...

# 1. Get the open-source TripoSG code (MIT license)
if [ ! -d TripoSG ]; then
  git clone --depth 1 https://github.com/VAST-AI-Research/TripoSG.git
fi

# 2. Python env (conda recommended; TripoSG targets Python 3.10)
#    conda create -n my3d python=3.10 -y && conda activate my3d

# 3. PyTorch for your CUDA version
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# 4. TripoSG deps + our small web app deps.
#    `diso` (mesh extraction) compiles CUDA code and needs torch visible while it
#    builds, so it is installed separately with --no-build-isolation.
#    It also needs the CUDA toolkit (nvcc), e.g.:  conda install -c nvidia cuda-toolkit
grep -v '^diso' TripoSG/requirements.txt > /tmp/my3d-req.txt
pip install -r /tmp/my3d-req.txt
pip install -r requirements-app.txt
pip install --no-build-isolation diso

echo "Done. Start the server with:  python server.py"
echo "Model weights (~several GB) download automatically on first start."
