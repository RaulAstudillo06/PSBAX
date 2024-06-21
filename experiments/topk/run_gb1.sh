#!/bin/bash
for acq_func in "bax" "ps" "random"
do
    python gb1_runner.py -s --policy $acq_func --first_trial 1 --trials 10 --max_iter 200 --batch_size 5 --model_type dkgp --epochs 10000 --k 10 &
done
wait