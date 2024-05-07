#!/bin/bash
for acq_func in "bax" "ps"
do
    python topk_runner.py -s --function original --dim 3 --trials 30 --max_iter 200 --batch_size 1 --len_path 400 --k 10 --policy $acq_func &
done
wait