#!/bin/bash
#SBATCH -J yolo-train
#SBATCH -p debug                 # <-- 确保这个分区有效
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --output=output_%j.log               # 输出日志文件 (%j 会替换为作业ID)

set -euo pipefail

# … 前略 …

# 激活/创建环境这段外侧关闭 -u
set +u
source ~/anaconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^yolov11s "; then
  conda create -n yolo-gpu117 python=3.10 -y
fi
conda activate yolo-gpu117
set -u
# 1) 进入工程目录
cd /public/home/yuyf/yuyf/yolo/ultralytics-main

# 2) 加载 conda（非交互脚本里必须）
source ~/anaconda3/etc/profile.d/conda.sh

# 3) 准备/激活环境（建议首次创建后，后续就走 else 分支，节省时间）
if ! conda env list | grep -q "^yolo-gpu117 "; then
  conda create -n yolo-gpu117 python=3.10 -y
fi
conda activate yolo-gpu117

# 4) 环境卫生与解析策略（避免系统 CUDA/用户 site 包干扰）
unset LD_LIBRARY_PATH
unset CUDA_HOME
unset CUDA_PATH
export PYTHONNOUSERSITE=1
conda config --env --set channel_priority strict
export CONDA_OVERRIDE_CUDA=11.7

# 5) 安装/修复 依赖（幂等，可反复执行）
# 5.1 先移除可能的 CPU 版 torch 及互斥包（无则跳过）
conda remove -y pytorch torchvision torchaudio pytorch-mutex cpuonly || true
pip uninstall -y torch torchvision torchaudio || true

# 5.2 固定 NumPy 到 1.x（与 torch 2.1.2 兼容）
conda install -y "numpy<2"

# 5.3 安装与驱动 515.57 匹配的 **CUDA 11.7** 组合（只用 pytorch + nvidia 两个频道）
conda install -y -c pytorch -c nvidia \
  pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.7

# 5.4 安装 Ultralytics 与 OpenCV（不让 pip 改 torch 依赖）
pip install --no-deps ultralytics==8.3.217
pip install --no-cache-dir opencv-python-headless

# 6) 自检（此处应看到 cuda=True 且 11.7，并打印 GPU 名称；NumPy 为 1.26.x）
echo "===== nvidia-smi ====="; nvidia-smi
echo "===== python & torch & numpy ====="
python - <<'PY'
import torch, sys, numpy as np
print("python:", sys.version)
print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available(), "| torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0))
print("numpy:", np.__version__)
PY

# 7) 开始训练（GPU 就绪时无需 device=cpu；若想先跑通可加 device=cpu）
python ./train.py
# 或：yolo train model=yolov8n.yaml data=data.yaml device=0