#!/bin/bash
for acq_func in "bax"
do
    for first_trial in 1 3 5 7 9
    do
        for batch_size in 1 5
        do
            python discobax_runner.py -s --problem_idx 3 --max_iter 200 --do_pca --pca_dim 5 --eta_budget 100 --batch_size $batch_size --policy $acq_func --first_trial $first_trial --trials 2 &
        done
    done 
done
wait