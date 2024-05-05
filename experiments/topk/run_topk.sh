#!/bin/bash
for acq_func in "bax" "ps"
do
    for batch_size in 1 3
    do
        python topk_runner.py -s --function himmelblau --dim 3 --max_iter 50 --trials 10 --batch_size $batch_size --policy $acq_func &
    done
done
wait