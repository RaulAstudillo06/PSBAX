#!/bin/bash
for acq_func in "bax" "ps"
do
    for samp_str in "mut" "cma"
    do
    # python california_runner.py -s --policy $acq_func
    python ackley_runner.py -s --dim 10 --max_iter 100 --trials 5 --samp_str $samp_str --policy $acq_func 
    done
done