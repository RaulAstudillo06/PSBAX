#!/bin/bash
for acq_func in "bax" "ps"
do
    python topk_runner.py -s --function original --use_mesh --steps 15 --dim 3 --trials 30 --max_iter 200 --batch_size 1 --policy $acq_func &
done
wait