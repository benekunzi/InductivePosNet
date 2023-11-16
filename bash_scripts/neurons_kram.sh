#!/bin/bash

for i in $(seq 1 5):
do
    python3 main.py --name 'neurons600_750' --learning_rate 0.005 --decay_rate 0.80 --n_layers 11 --distance 750 --n_neurons 600
    python3 main.py --name 'neurons900_750' --learning_rate 0.005 --decay_rate 0.80 --n_layers 11 --distance 750 --n_neurons 900
done