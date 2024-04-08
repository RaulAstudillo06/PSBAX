#!/bin/bash
for acq_func in "bax" "ps"
do
    for samp_str in "mut"
    do
    # python california_runner.py -s --policy $acq_func
    python hartmann_runner.py -s --max_iter 200 --trials 5 --samp_str $samp_str --policy $acq_func 
    done
done