#!/bin/bash
for first_trial in 1 6
do
    for acq_func in "bax" "ps"
    do
        for samp_str in "mut" "cma"
        do
        python hartmann_runner.py -s --max_iter 100 --trials 5 --samp_str $samp_str --policy $acq_func --first_trial $first_trial --batch_size 5 &
        done
    done
done
wait