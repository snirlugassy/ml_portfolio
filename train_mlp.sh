#!/bin/bash
depths=(5 7 10);
lrs=(0.0001 0.001 0.005);

for norm_factor in $(seq 0.0 0.1 0.4); do
for learning_rate in ${lrs[@]}; do
for depth in ${depths[@]}; do

    python -m mlp.train_mlp \
        --seed=42 \
        --epochs=150 \
        --batch_size=256 \
        --pct-change=1 \
        --depth=$depth \
        --dropout=0.05 \
        --normalization-factor=$norm_factor \
        --learning-rate=$learning_rate \
        --step-lr-gamma=0.95 \
        --step-lr-every=15 \
        --use_batch_norm \
        --use_skip_connection \
        --apply_augments \
        --device=cuda:0

    python -m mlp.train_mlp \
        --seed=42 \
        --epochs=150 \
        --batch_size=256 \
        --pct-change=1 \
        --depth=$depth \
        --dropout=0.05 \
        --normalization-factor=$norm_factor \
        --learning-rate=$learning_rate \
        --step-lr-gamma=0.95 \
        --step-lr-every=15 \
        --apply_augments \
        --device=cuda:0

done
done
done
