
for depth in $(seq 6 9); do
    for dropout in $(seq 0.0 0.1 0.3); do
        for learning_rate in $(seq 0.0001 0.0002 0.0009); do 
            python -m mlp.train_mlp \
                --seed=42 \
                --epochs=50 \
                --depth=$depth \
                --dropout=$dropout \
                --learning-rate=$learning_rate \
                --use_batch_norm \
                --apply_augments \
                --device=cuda:0
        done
    done
done

# --use_skip_connection \