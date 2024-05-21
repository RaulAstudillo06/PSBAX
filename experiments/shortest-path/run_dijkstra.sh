#!/bin/bash
for acq_func in "ps" "random" "bax"
do
    python new_california_runner.py -s --trials 30 --policy $acq_func &
done
wait