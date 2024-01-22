#!/bin/bash

#SBATCH --job-name=ChatGPT-42
#SBATCH --partition=p1
#SBATCH --time=0-12:00:00

# DGX features 10 threads and 64 GB memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G

#SBATCH --array=1-10
CUDA_VISIBLE_DEVICES="" python3 main.py --name 'mlp-episodisch-100-fixedScaling-SLR0025_Neu900' --learning_rate 0.005 --n_neurons 900 --decay_rate 0.80 --n_layers 11 --distance 750 --batch_size 1024 --epochs 2880 --regularization True --regularization_value 0.00000001 --dropout True --dropout_value 0.01 --activation_final "tanh"