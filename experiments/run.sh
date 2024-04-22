#!/bin/bash
for acq_func in "ps" "bax" "random"
do
    # python california_runner.py -s --policy $acq_func
    # python multiobjective_runner.py -s --policy $acq_func --problem dtlz2 --n_dim 10 --n_obj 3 --n_gen 500 --pop_size 40 --max_iter 50 --n_init 100 --batch_size 5
    python multiobjective_runner.py -s --policy $acq_func --problem zdt2 --n_dim 6 --n_obj 2 --n_gen 50 --pop_size 10 --max_iter 50 --n_init 100 --batch_size 5
    # python multiobjective_runner.py -s --policy $acq_func --problem dtlz1 --batch_size 5 --n_dim 6 --n_obj 2 --n_gen 50 --pop_size 10 --max_iter 50 --n_init 20
    # python multiobjective_runner.py -s --policy $acq_func --problem dtlz2 --n_dim 3 --n_obj 2 --n_gen 50 --pop_size 10 --max_iter 30 --n_init 10 
    done 
done 
# Check for weird characters: cat -v run.sh
# Remove line breaks: tr -d '\r' < run.sh > fixedrun.sh