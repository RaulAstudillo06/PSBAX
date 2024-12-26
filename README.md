# Practical Bayesian Algorithm Execution via Posterior Sampling

> [Practical Bayesian Algorithm Execution via Posterior Sampling](https://arxiv.org/abs/2410.20596)\
> Chu Xin Cheng, Raul Astudillo, Thomas Desautels, Yisong Yue\
> Advances in Neural Information Processing Systems, 2024


## Environment setup
This project uses conda to manage the environment. Clone the repository and navigate to the root directory.
```
$ git clone https://github.com/RaulAstudillo06/PSBAX.git
$ cd PSBAX
```
To set the environment, run the following command:
```
$ conda env create -f env.yml
```
Activate the environment:
```
$ conda activate PSBAX
```

## Demos
In the `demos` directory, there are three demo files that can be run to showcase the capabilities of PSBAX.

- Demo 1: Local Bayesian Optimization
    ```
    $ python local_bo.py
    ```
- Demo 2: Level Set Estimation
    ```
    $ python level-set.py
    ```
- Demo 3: Top-k Estimation
    ```
    $ python topk.py
    ```

The experiment results are saved in the `demos/results` directory. A graph of the results is automatically generated and saved in the `demos/plots` directory.


## Main experiments

The `experiments` directory contains the runner files used to run the experiments conducted in the main paper.

### DiscoBAX
The DiscoBAX experiments were performed using data from the paper "DiscoBAX - Discovery of optimal intervention sets in genomic experiment design" by Lyle et al. The data included in this repository is preprocessed and saved in the `discobax/data` directory. To access the full dataset, please refer to the original DiscoBAX repository [here](https://github.com/amehrjou/DiscoBAX). 

To run the experiments for DiscoBAX, navigate to the `experiments/discoBAX` directory and run the following command:
```
$ bash run_discobax.sh discobax_runner.py
```

### Local Bayesian Optimization
To run the experiments for Local Bayesian Optimization, navigate to the `experiments/single-objective` directory and run the following commands:

- Ackley 10D synthetic function:
    ```
    $ bash run_local-bo.sh ackley_runner.py
    ```
- Hartmann 6D synthetic function:
    ```
    $ bash run_local-bo.sh hartmann_runner.py
    ```

### Level Set Estimation
To run the experiments for Level Set Estimation, navigate to the `experiments/level-set` directory and run the following command:
```
$ bash run_level-set.sh
```

### Top-k Estimation
To run the experiments for Top-k Estimation, navigate to the `experiments/topk` directory and run the following commands:

- Rosenbrock synthetic function:
    ```
    $ bash run_topk.sh topk_runner.py
    ```
- GB1 protein fitness:
    ```
    $ bash run_topk.sh gb1_runner.py
    ```

## Citation
```
@inproceedings{neiswanger2021bayesian,
  title         = {Practical Bayesian Algorithm Execution via Posterior Sampling},
  author        = {Cheng, Chu Xin and Astudillo, Raul and Desautels, Thomas and Yue, Yisong},
  booktitle={Advances in Neural Information Processing Systems},
  volume = {38},
  publisher = {Curran Associates, Inc.},
  year={2024}
}
