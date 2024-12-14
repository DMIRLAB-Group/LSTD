d={
    'Traffic':{
        1: ' --method LSTD --root_path ./data/ --data Traffic --n_inner 1 --test_bsz 1 --features M --seq_len 60 --label_len 0 '
             ' --des "Exp" --itr 1 --train_epochs 6  --online_learning "full" --L1_weight 0.001 --dropout 0'
             ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 1 '
             ' --depth 10 --hidden_dim 512 --hidden_layers 1 --tau 0.7 --batch_size 32 --learning_rate 0.003 --mode time',
        24: ' --method LSTD --root_path ./data/ --data Traffic --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
             ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0 '
             ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 24 '
             ' --depth 9 --hidden_dim 512 --hidden_layers 2 --tau 0.75 --batch_size 16 --learning_rate 0.003 --mode time',
        48: ' --method LSTD --root_path ./data/ --data Traffic --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
             ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0 '
             ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 48 '
             ' --depth 10 --hidden_dim 512 --hidden_layers 2 --tau 0.7 --batch_size 16 --learning_rate 0.003 --mode time'
    },


    'Exchange':{
            1: ' --method LSTD --root_path ./data/ --data Exchange --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
                 ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
                 ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 1 '
                 ' --depth 10 --hidden_dim 448 --hidden_layers 1 --tau 0.7 --batch_size 4 --learning_rate 0.003 --mode time',
            24: ' --method LSTD --root_path ./data/ --data Exchange --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
                 ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
                 ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 24 '
                 ' --depth 10 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.003 --mode time',
            48: ' --method LSTD --root_path ./data/ --data Exchange --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
                 ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
                 ' --L2_weight 0.001  --zd_kl_weight 0.0001  --zc_kl_weight 0.0001 --pred_len 48 '
                 ' --depth 9 --hidden_dim 448 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.003 --mode time'
        },

    'WTH': {
        1: ' --method LSTD --root_path ./data/ --data WTH --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
             ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
             ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 1 '
             ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.003 --mode time',
        24: ' --method LSTD --root_path ./data/ --data WTH --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
              ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
              ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 24 '
              ' --depth 9 --hidden_dim 256 --hidden_layers 1 --tau 0.7 --batch_size 4 --learning_rate 0.002 --mode var',
        48: '--method LSTD --root_path ./data/ --data WTH --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
              '--des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
              '--L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 48 '
             ' --depth 9 --hidden_dim 256 --hidden_layers 1 --tau 0.7 --batch_size 4 --learning_rate 0.001 --mode var'
    },

    'ECL': {
        1: ' --method LSTD --root_path ./data/ --data ECL --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
             ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
             ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 1 '
            ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 16 --learning_rate 0.002 --mode time',
        24: ' --method LSTD --root_path ./data/ --data ECL --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
              ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
              ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 24 '
              ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.002 --mode var',
        48: ' --method LSTD --root_path ./data/ --data ECL --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
              ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
              ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 48 '
                ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.002 --mode var'
    },

    'ETTh2': {
        1: ' --method LSTD --root_path ./data/ --data ETTh2 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
             ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
             ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 1 '
             ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 4 --learning_rate 0.003 --mode time',
        24: ' --method LSTD --root_path ./data/ --data ETTh2 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
              ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
              ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 24 '
            ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 4 --learning_rate 0.003 --mode var',
        48: ' --method LSTD --root_path ./data/ --data ETTh2 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
              ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
              ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 48 '
                ' --depth 9 --hidden_dim 256 --hidden_layers 2 --tau 0.7 --batch_size 32 --learning_rate 0.003 --mode var'


    },

    'ETTm1': {
        1: ' --method LSTD --root_path ./data/ --data ETTm1 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
             ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
             ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 1 '
            ' --depth 9 --hidden_dim 512 --hidden_layers 1 --tau 0.75 --batch_size 8 --learning_rate 0.003 --mode time',
        24: ' --method LSTD --root_path ./data/ --data ETTm1 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
              ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
              ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 24 '
              ' --depth 9 --hidden_dim 256 --hidden_layers 1 --tau 0.7 --batch_size 4 --learning_rate 0.003 --mode var',
        48: ' --method LSTD --root_path ./data/ --data ETTm1 --n_inner 1 --test_bsz 1  --features M --seq_len 60 --label_len 0 '
              ' --des "Exp" --itr 1 --train_epochs 6 --online_learning "full" --L1_weight 0.001 --dropout 0'
              ' --L2_weight 0.001  --zd_kl_weight 0.001  --zc_kl_weight 0.001 --pred_len 48 '
               ' --depth 9 --hidden_dim 256 --hidden_layers 1 --tau 0.7 --batch_size 4 --learning_rate 0.002 --mode var'
    },

}