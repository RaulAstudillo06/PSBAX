#!/bin/bash
for acq_func in "bax" "ps" "random"
do
    python topk_runner.py -s --function himmelblau --dim 2 --trials 10 --max_iter 100 --batch_size 1 --len_path 200 --k 4 --policy $acq_func &
done
wait