
for depth in $(seq 5 7); do
    for dropout in $(seq 0.2 0.1 0.4); do
        # for learning_rate in $(seq 0.0001 0.0002 0.0009); do 
            python -m mlp.train_mlp \
                --seed=42 \
                --epochs=100 \
                --batch_size=128 \
                --depth=$depth \
                --dropout=$dropout \
                --learning-rate=0.00005 \
                --use_batch_norm \
                --apply_augments \
                --device=cuda:0 \
        # done
    done
done

# --use_skip_connection \