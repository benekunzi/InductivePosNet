#!/bin/bash

#SBATCH --job-name=ChatGPT-42
#SBATCH --partition=p1
#SBATCH --time=0-12:00:00

# DGX features 10 threads and 64 GB memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G

#SBATCH --array=1-10
python3 main.py --name "lstm_first_test"