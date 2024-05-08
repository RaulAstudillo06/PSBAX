#!/bin/bash
for first_trial in 1 6
do
    for acq_func in "bax" "ps"
    do
        for batch_size in 3 5
        do
        python ackley_runner.py -s --max_iter 100 --dim 10 --trials 5 --policy $acq_func --first_trial $first_trial --batch_size $batch_size &
        done
    done
done
wait