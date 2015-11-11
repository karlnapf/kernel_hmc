from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian
from kernel_hmc.proposals.kmc import KMCStatic
import numpy as np


def get_target_momentum():
    D = 2
    target = IsotropicZeroMeanGaussian(D=D)
    momentum = IsotropicZeroMeanGaussian(D=D)
    
    return target, momentum

def get_hmc_parameters():
    num_steps_min = 1
    num_steps_max = 10
    step_size_min = 0.01
    step_size_max = 0.1
    
    return num_steps_min, num_steps_max, step_size_min, step_size_max

def get_static_surrogate(D):
    N = 200
    X = np.random.randn(N, D)
    est = KernelExpLiteGaussian(sigma=1, lmbda=.1, D=D)
    est.fit(X)
    
    return est

def test_kmc_base_init_execute():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, momentum = get_target_momentum()
    surrogate = get_static_surrogate(momentum.D)
    KMCStatic(surrogate, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)

