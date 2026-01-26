#!/bin/bash
#!/usr/bin/env python3

#SBATCH --job-name=unet                      # 作业名称
#SBATCH --output=output_%j.log               # 输出日志文件 (%j 会替换为作业ID)
#SBATCH --partition=debug                    # 指定分区
#SBATCH --gres=gpu:1                        # 请求1个GPU
#SBATCH --ntasks=1                          # 启动1个任务
#SBATCH --cpus-per-task=2                   # 为任务分配4个CPU核心
#SBATCH --mem=128G                          # 分配128GB内存


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
#source ~/anaconda3/etc/profile.d/conda.sh
# 运行训练脚本
#python ./write.py
#python  ./resize1.py
#python ./huidu.py
#python ./tran.py
#python ./cuda.py
#python ./train.py
#python ./trainasff.py
#python ./trainafpn3.py
python ./trainafpn4.py
nvidia-smi
python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("python:", sys.version)
PY
#python ./evaluate.py
#python predict.py --model ./checkpoints_optimized2/best_checkpoint.pth --input ./data/plantsegv2/images/train_res/apple_black_rot_1.jpg --output test_out9.png
#python test.py --model ./checkpoints_optimized/best_checkpoint.pth --test_data ./data/plantsegv2 --classes 1 --batch-size 4 --scale 1.0
#python datasplit.py --data_dir ./data --val_percent 0.2
#python ./diceoone.py
#python ./dicedif.py
#python ./predictall.py --model ./checkpoints_optimized2/best_checkpoint.pth --input_dir ./data/plantsegv2/images/train_res/ --output_dir ./output_maskstr