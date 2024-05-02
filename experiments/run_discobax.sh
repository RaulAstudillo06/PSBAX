#!/bin/bash
pidfile="run_discobax_pids.txt"  # File where PIDs will be stored

# Clean the file at the start of the script
> "$pidfile"

for acq_func in "OPT" "ps" "bax"
do
    for first_trial in 1 3 5 7 9
    do
        python discobax_runner.py -s --problem_idx 3 --max_iter 200 --do_pca --pca_dim 5 --eta_budget 100 --batch_size 5 --policy $acq_func --first_trial $first_trial --trials 2 &
        echo $! >> "$pidfile"  # Store the PID of the last background job
    done
done
wait