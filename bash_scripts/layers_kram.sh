#!/bin/bash

for i in $(seq 1 10):
do
    python3 main.py --name 'layers_8_750' --learning_rate 0.005 --decay_rate 0.80 --n_layers 8 --distance 750
    python3 main.py --name 'layers_9_750' --learning_rate 0.005 --decay_rate 0.80 --n_layers 9 --distance 750
    python3 main.py --name 'layers_10_750' --learning_rate 0.005 --decay_rate 0.80 --n_layers 10 --distance 750
    python3 main.py --name 'layers_11_750' --learning_rate 0.005 --decay_rate 0.80 --n_layers 11 --distance 750
done