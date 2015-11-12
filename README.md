# Kernel Hamiltonian Monte Carlo

[![Build Status](https://travis-ci.org/karlnapf/kernel_hmc.png)](https://travis-ci.org/karlnapf/kernel_hmc)
[![Coverage Status](https://coveralls.io/repos/karlnapf/kernel_hmc/badge.svg?branch=master&service=github)](https://coveralls.io/github/karlnapf/kernel_hmc?branch=master)

Code for NIPS 2015 [Gradient-free Hamiltonain Monte Carlo with Efficient Kernel Exponential Families](http://arxiv.org/abs/1506.02564).

This package implements the kernel HMC part of the paper. It heavily depends on the [kernel exponential family package](https://github.com/karlnapf/kernel_exp_family), where all gradient estimation code is located.

My [blog post](http://herrstrathmann.de/kamiltonian-monte-carlo/) about KMC.

An [IPython notebook](http://nbviewer.ipython.org/gist/karlnapf/da0089726c43ed52a899) featuring KMC lite's ability to move in previously unexplored regions.



Install dependencies:

    pip install -r https://raw.githubusercontent.com/karlnapf/kernel_hmc/master/requirements.txt

Optional dependencies are:
 * ```[cholupdate](https://github.com/jcrudy/choldate)``` for effecient low-rank updates of Cholesky factors of covariances. Speeds up Adaptive-Metropolis and KMC Finite from cubic to quadratic costs, see paper.
 * ```[Shogun-toolbox](https://github.com/shogun-toolbox/shogun)``` for the Gaussian Process marginal posterior over hyper-parameters example. To compute unbiased estimates of the marginal likelihood via approximate inference and importance sampling.
 * ```[theano](https://github.com/theano/theano)``` for the Banana example, to compute gradients via auto-grad.
    
Install ```kernel_hmc```:

    pip install git+https://github.com/karlnapf/kernel_hmc.git

A list of examples can be found [here](kernel_hmc/examples). For example, run

    python -m kernel_hmc.examples.demo_trajectories.py
    python -m kernel_hmc.examples.demo_mcmc_kmc_static.py

