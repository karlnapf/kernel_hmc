# Kernel Hamiltonian Monte Carlo

[![Build Status](https://travis-ci.org/karlnapf/kernel_hmc.png)](https://travis-ci.org/karlnapf/kernel_hmc)
[![Coverage Status](https://coveralls.io/repos/karlnapf/kernel_hmc/badge.svg?branch=master&service=github)](https://coveralls.io/github/karlnapf/kernel_hmc?branch=master)

Code for NIPS 2015 [Gradient-free Hamiltonain Monte Carlo with Efficient Kernel Exponential Families](http://arxiv.org/abs/1506.02564).

This package implements the kernel HMC part of the paper. It heavily depends on the [kernel exponential family package](https://github.com/karlnapf/kernel_exp_family), where all gradient estimation code is located.

Install dependencies:

    pip install -r https://raw.githubusercontent.com/karlnapf/kernel_hmc/master/requirements.txt
    
Install ```kernel_hmc```:

    pip install git+https://github.com/karlnapf/kernel_hmc.git

A list of examples can be found [here](kernel_hmc/examples). For example, run

    python -m kernel_hmc.examples.demo_simple.py

