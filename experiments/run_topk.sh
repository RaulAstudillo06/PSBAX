#!/bin/bash
for acq_func in "bax" "ps"
do
    python topk_runner.py -s --function original --dim 3 --max_iter 200 --trials 10 --policy $acq_func 
done