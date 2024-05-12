#!/bin/bash
for acq_func in "bax" "ps" "random"
do
    python new_california_runner.py -s --trials 10 --policy $acq_func &
done
wait