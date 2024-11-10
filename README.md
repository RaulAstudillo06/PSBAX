# PSBAX

## Link to Paper: [Practical Bayesian Algorithm Execution via Posterior Sampling](https://arxiv.org/abs/2410.20596)

## Environment Setup
This project uses conda to manage the environment. Clone the repository and navigate to the root directory.
```
$ git clone https://github.com/RaulAstudillo06/PSBAX.git
$ cd PSBAX
```
To setup the environment, run the following command:
```
$ conda env create -f env.yml
```
Activate the environment:
```
$ conda activate PSBAX
```

## Running Demos
In the `demos` directory, there are three demos files that can be run to showcase the capabilities of PSBAX.

- Demo 1: Local Bayesian Optimization
    ```
    $ python local_bo.py
    ```
- Demo 2: Level Set Estimation
    ```
    $ python level-set.py
    ```
- Demo 3: Top-k
    ```
    $ python topk.py
    ```

The experiment results are saved in the `demos/results` directory. A graph of the results is automatically generated and saved in the `demos/plots` directory.


## Running the Experiments

The `experiments` directory contains the runner files used to run the experiments conducted for the paper. The DiscoBAX 

### DiscoBAX
The DiscoBAX experiments were ran using data from the paper "DiscoBAX - Discovery of optimal intervention sets in genomic experiment design" by Lyle et al. The data included in this repository is preprocessed and saved in the `discobax/data` directory. To access the full dataset, please refer to the original DiscoBAX repository [here](https://github.com/amehrjou/DiscoBAX). 

To run the experiments for DiscoBAX, navigate to the `experiments/discoBAX` directory and run the following commands:
```
# for GP with PCA for dimensionality reduction:
$ bash run_discobax.sh discobax_runner.py

# for using Deep Kernel Learning:
$ bash run_discobax.sh discobax_dkl_runner.py
```

