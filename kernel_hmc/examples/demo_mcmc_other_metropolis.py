from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian
from kernel_hmc.examples.plotting import visualise_trace
from kernel_hmc.mini_mcmc.mini_mcmc import mini_mcmc
from kernel_hmc.proposals.metropolis import StandardMetropolis,\
    AdaptiveMetropolis, KernelAdaptiveMetropolis
from kernel_hmc.tools.log import Log
import matplotlib.pyplot as plt
import numpy as np


# banana gradient depends on theano, which is an optional dependency
try:
    from kernel_hmc.densities.banana import Banana
    banana_available = True
except ImportError:
    banana_available = False

Log.set_loglevel(20)

if __name__ == '__main__':
    """
    This example shows how to run all Metropolis-Hastings sampler (including
    the Kernel Adaptive Metropolis-Hastings) on a simple target.
    """
    # possible to change
    D = 3
    N = 1000
    
    # target is banana density, fallback to Gaussian if theano is not present
    if banana_available:
        target = Banana(D=D)
    else:
        target = IsotropicZeroMeanGaussian(D=D)

    samplers = [
                StandardMetropolis(target, D),
                AdaptiveMetropolis(target, D),
                KernelAdaptiveMetropolis(target, D, N=200)
                
                ]

    for sampler in samplers:
        # MCMC parameters, feel free to increase number of iterations
        start = np.zeros(D)
        num_iter = 1000
        
        # run MCMC
        samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(sampler, start, num_iter, D)
        
        visualise_trace(samples, log_pdf, accepted, step_sizes)
        plt.suptitle("%s, acceptance rate: %.2f" % \
                     (sampler.__class__.__name__, np.mean(accepted)))
        
    plt.show()
