#!/bin/bash

#SBATCH --job-name=ChatGPT-42
#SBATCH --partition=p1
#SBATCH --time=0-12:00:00

# DGX features 10 threads and 64 GB memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G

#SBATCH --array=1-10
python3 main.py --name "lstm_batch_2048" --batch_size 2048
python3 main.py --name "lstm_batch_1024" --batch_size 1024 
python3 main.py --name "lstm_batch_512" --batch_size 512 
python3 main.py --name "lstm_batch_256" --batch_size 256 
python3 main.py --name "lstm_batch_128" --batch_size 128 
python3 main.py --name "lstm_batch_64" --batch_size 64 