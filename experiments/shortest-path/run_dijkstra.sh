#!/bin/bash
for acq_func in "bax" "ps"
do
    python dijkstra_runner.py -s --trials 10 --policy $acq_func &
done
wait