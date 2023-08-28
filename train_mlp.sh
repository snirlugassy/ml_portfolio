#!/bin/bash

depths=(7 9 11);
lrs=(0.00001 0.0001 0.0005);

for learning_rate in ${lrs[@]}; do
    for depth in ${depths[@]}; do

        python -m mlp.train_mlp_v2 \
            --output-dir=/home/snirlugassy/ml_portfolio/experiments/mlp_v2/ \
            --seed=42 \
            --epochs=200 \
            --batch_size=256 \
            --pct-change=1 \
            --depth=$depth \
            --dropout=0.05 \
            --normalization-factor=0.0 \
            --learning-rate=$learning_rate \
            --step-lr-gamma=0.9 \
            --step-lr-every=15 \
            --use_batch_norm \
            --apply_augments \
            --device=cuda:0

    done
done
