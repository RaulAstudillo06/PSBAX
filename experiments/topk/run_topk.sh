#!/bin/bash
FILENAME=$1

for acq_func in "bax" "ps" "random"
do
    python $FILENAME -s --policy $acq_func
done

