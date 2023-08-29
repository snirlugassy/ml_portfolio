#!/bin/bash

depths=(5 7 9 11);
lrs=(1e-3 1e-4 1e-5);
normalization_factors=(0.0 0.1 0.5);
dropouts=(0.0 0.2 0.4);
seeds=(33 42 56);

for learning_rate in ${lrs[@]}; do
for depth in ${depths[@]}; do
for nf in ${normalization_factors[@]}; do
for dropout in ${dropouts[@]}; do
for seed in ${seeds[@]}; do

        # with skip connection
        python train_mlp_v1.py \
            --output-dir=/home/snirlugassy/ml_portfolio/experiments/mlp_v1/ \
            --seed=$seed \
            --epochs=200 \
            --batch_size=256 \
            --pct-change=1 \
            --depth=$depth \
            --dropout=$dropout \
            --normalization-factor=$nf \
            --learning-rate=$learning_rate \
            --step-lr-gamma=0.9 \
            --step-lr-every=15 \
            --use_batch_norm \
            --use_skip_connection \
            --apply_augments \
            --device=cuda:0

        # without skip connection
        python train_mlp_v1.py \
            --output-dir=/home/snirlugassy/ml_portfolio/experiments/mlp_v1/ \
            --seed=$seed \
            --epochs=200 \
            --batch_size=256 \
            --pct-change=1 \
            --depth=$depth \
            --dropout=$dropout \
            --normalization-factor=$nf \
            --learning-rate=$learning_rate \
            --step-lr-gamma=0.9 \
            --step-lr-every=15 \
            --use_batch_norm \
            --apply_augments \
            --device=cuda:0

done
done
done
done
done
