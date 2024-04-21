#!/bin/bash
for acq_func in "ps" "bax" "OPT"
do
    python discobax_runner.py -s --problem_idx 3 --num_iter 150 --do_pca --pca_dim 10 --data_size 1700 --use_top --n_init 10 --eta_budget 100 --policy $acq_func --first_trial 1 --last_trial 5 
done