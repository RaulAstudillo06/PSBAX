


# Multiobjective
- DTLZ2 with noise
- ZDT2 with batch size = 5, 10

# Singleobjective
- Ackley
    - Batch size
        - [x] 10D with different batch sizes
    - Noise 
- Hartmann 
    - Batch size
        - [ ] 6D with different batch sizes
    - Noise


- Find process id
```
tmux ls  -F 'socket_path: #{socket_path} | session_name: #{session_name} | server_pid: #{pid} | pane_pid: #{pane_pid}'
```

- Terminate job
    - First ctrl + Z
```
kill -SIGTERM %1

# can check by 
pidof 1
```



------

Experiments to run:
- Local BO
    - Hartmann
        ```
        python hartmann_runner.py -s --trials 30 --policy ps
        python hartmann_runner.py -s --trials 30 --policy bax
        ```
    - Rastrigin
        ```
        python rastrigin_runner.py -s --dim 5 --trials 30 --policy ps
        python rastrigin_runner.py -s --dim 5 --trials 30 --policy bax
        ```
- Topk 
    - Himmelblau
        ```python topk_runner.py -s --trials 30 --policy random```
- Dijkstra
    - 
- DiscoBAX
    - Rerun experiments so that the objective is fixed
    - `problem[3] = sanchez_2021_tau_top_1700`
        ```
        python discobax_runner.py -s --problem_idx 3 --num_iter 100 --do_pca True --pca_dim 5 --data_size 1700 --eta_budget 100 --policy bax --first_trial 1 --last_trial 30
        ```






- `topk` has output as a Namespace
    - performance metric needs output.y


--- 
# TODO
- `dkgp`: comment out outcome transform

- what is the architecture that we want to use?
    - what the NN architecture is

- Compare posterior sampling w/ random
    - Do we know how good the PCA is? How much information have we lost?
    - Singular values -> check dimensionality
    - check y values of dataset -> is it sparse
    
- Check how good the fit is

- Try evaluate everything in the output

--- 
# Implementation

- in `algo.run_algorithm_on_f`, `self.initialize()` clears the Namespace for `self.exe_path`

- Dataset file names
    ```[
        "schmidt_2021_ifng", # 17465
        "schmidt_2021_il2", # 17465
        "zhuang_2019", # 17528
        "sanchez_2021_tau", # 17176
        "zhu_2021_sarscov2_host_factors", # 16670
    ]
    ```
    - `test_schmidt_2021_ifng.csv` : (1746, 21)

- Standardize algo.execute, algo.run_algorithm_on_f

- algo.execute should always return a np.array of output

## Questions
- `discobax_trial`
    - Performance Metric and Algo both get a dataset
