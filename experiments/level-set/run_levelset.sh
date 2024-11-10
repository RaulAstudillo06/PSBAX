#!/bin/bash
for acq_func in "ps" "bax" "random" "lse"
do
    for problem in "volcano" "himmelblau"
    python levelset_runner.py -s --problem $problem --trials 30 --policy $acq_func
done
