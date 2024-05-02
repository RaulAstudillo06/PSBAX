#!/bin/bash
for first_trial in 1 6
do
    # Run the command in the background
    python multiobjective_runner.py -s --policy random --problem zdt2 --noise 0.1 --n_dim 6 --n_obj 2 --n_gen 500 --pop_size 100 --max_iter 50 --batch_size 5 --first_trial $first_trial & 
    # python multiobjective_runner.py -s --policy random --problem dtlz2 --noise 0 --n_dim 6 --n_obj 2 --n_gen 500 --pop_size 100 --max_iter 50 --batch_size 5 --first_trial $first_trial &
done
# Wait for all background jobs to finish
wait
