#!/bin/bash
for acq_func in "ps" "bax" "random"
do
    python levelset_runner.py -s --trials 10 --policy $acq_func &
done
wait