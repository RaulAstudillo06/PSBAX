#!/bin/bash

for acq_func in "random" "ps" "bax"
do  
    for problem_idx in 0 1
    do 
        python discobax_runner.py -s --problem_idx $problem_idx --do_pca --pca_dim 5 --eta_budget 100 --data_size 5000 --batch_size 1 --policy $acq_func --trials 30
    done
done
wait