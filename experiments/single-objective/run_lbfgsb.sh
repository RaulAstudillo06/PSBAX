#!/bin/bash
for first_trial in 1 3 5 7 9
do
    for acq_func in "bax" "ps"
    do
        for batch_size in 1 5
        do
            python lbfgsb_runner.py -s --max_iter 100 --trials 2 --policy $acq_func --first_trial $first_trial --batch_size $batch_size & 
        done
    done
done
wait