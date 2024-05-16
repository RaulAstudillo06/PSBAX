#!/bin/bash
for acq_func in "ps" "bax" "random"
do
    python levelset_runner.py -s --problem volcano --trials 30 --policy $acq_func --max_iter 50 &
done
wait