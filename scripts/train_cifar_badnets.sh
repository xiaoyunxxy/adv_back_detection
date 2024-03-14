#!/bin/bash -e
#SBATCH --job-name cifar_benign
#SBATCH --partition=icis
#SBATCH --account=icis
#SBATCH --qos=icis-preempt
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=60:00:00
#SBATCH --output=./slurm_log/my-experiment-%j.out
#SBATCH --error=./slurm_log/my-experiment-%j.err
#SBATCH --mail-user=xiaoyun.xu@ru.nl
#SBATCH --mail-type=BEGIN,END,FAIL

source /ceph/dis-ceph/xxu/pytorch/bin/activate
cd /home/xxu/b_detection

CUDA_VISIBLE_DEVICES=1 python train_backdoor_cifar.py --output-dir './bd_exp/train_cifar_badnets' --poison-type 'badnets'