#!/bin/bash

FILENAME=$1

for acq_func in "random" "ps" "bax"
do  
    for problem_idx in 0 1
    do 
        python $FILENAME -s --problem_idx $problem_idx --eta_budget 100 --data_size 5000 --batch_size 1 --policy $acq_func --trials 30
    done
done

