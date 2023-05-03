#!/bin/bash
#SBATCH --job-name=testrjob
#SBATCH --partition=gpu
#SBATCH --mail-user=kartik.anand.19031@iitgoa.ac.in
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --no-kill
#SBATCH --output=job_output/newjob_%j.out
##SBATCH --cpus-per-gpu=40

cd /home/kartik.anand.19031/text_spotting/testr/TESTR
echo $SLURM_JOB_NODELIST > job_output/hostfile_$SLURM_JOBID
nvidia-smi
nvcc --version
#module load cuda/11.3
module load anaconda3/2021.11
conda init --all
source ~/.bashrc
#cat ~/.bash_profile > bash_profile
#cat ~/.bashrc > bashrc
conda activate testr
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
#python check_gpu.py
#python tools/train_net.py --config-file configs/TESTR/ICDAR15/TESTR_R_50_Polygon.yaml --num-gpus 1

## VISUALIZE:
#python demo/demo.py --config-file configs/TESTR/ICDAR15/TESTR_R_50_Polygon.yaml --input TESTR_HOME/datasets/icdar2015/test_images --output vis_out/ --opts MODEL.WEIGHTS $TESTR_HOME/output/TESTR/icdar15/TESTR_R_50_Polygon_Lite/model_0000999.pth MODEL.TRANSFORMER.INFERENCE_TH_TEST 0.3

## EVALUATE:
#python tools/train_net.py --config-file configs/TESTR/ICDAR15/TESTR_R_50_Polygon.yaml --eval-only MODEL.WEIGHTS $TESTR_HOME/output/TESTR/icdar15/TESTR_R_50_Polygon_Lite/model_final.pth
