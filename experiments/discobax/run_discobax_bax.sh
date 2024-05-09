#!/bin/bash
pidfile="discobax_pids_bax.txt"  # File where PIDs will be stored

# Clean the file at the start of the script
> "$pidfile"

for acq_func in "bax"
do
    for first_trial in 1
    do
        for batch_size in 1 5
        do
            python discobax_runner.py -s --problem_idx 0 --max_iter 100 --do_pca --pca_dim 5 --n_init 100 --eta_budget 100 --data_size 5000 --batch_size $batch_size --policy $acq_func --first_trial $first_trial --trials 10 &
            echo $! >> "$pidfile"  # Store the PID of the last background job
        done
    done
done
wait 
