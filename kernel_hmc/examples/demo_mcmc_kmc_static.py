from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian, \
    sample_gaussian
from kernel_hmc.examples.plotting import visualise_trace
from kernel_hmc.mini_mcmc.mini_mcmc import mini_mcmc
from kernel_hmc.proposals.kmc import KMCStatic
import matplotlib.pyplot as plt
import numpy as np


# banana gradient depends on theano, which is an optional dependency
try:
    from kernel_hmc.densities.banana import Banana, sample_banana
    banana_available = True
except ImportError:
    banana_available = False

if __name__ == '__main__':
    # for D=2, the fitted log-density is plotted, otherwise trajectory only
    D = 2
    N = 1000
    
    # target is banana density, fallback to Gaussian if theano is not present
    if banana_available:
        target = Banana(D=D)
        X = sample_banana(N, D)
    else:
        target = IsotropicZeroMeanGaussian(D=D)
        X = sample_gaussian(N=N)

    # plot trajectories for both KMC lite and finite, parameters are chosen for D=2
    for surrogate in [
                        KernelExpFiniteGaussian(gamma=.5, lmbda=0.001, m=N, D=D),
                        KernelExpLiteGaussian(sigma=20., lmbda=0.001, D=D)
                      ]:
        surrogate.fit(X)
        
        # HMC parameters, fixed here, use oracle mean variance to set momentum
        momentum = IsotropicZeroMeanGaussian(D=D, sigma=np.sqrt(np.mean(np.var(X, 0))))
        num_steps_min = 100
        num_steps_max = 100
        step_size_min = 1.
        step_size_max = 1.
        
        # kmc sampler instance
        kmc = KMCStatic(surrogate, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)
        
        # MCMC parameters
        start = X[0]
        num_iter = 300
        
        # run MCMC
        samples, proposals, accepted, acc_prob, log_pdf, times = mini_mcmc(kmc, start, num_iter, D)
        
        visualise_trace(samples, log_pdf, accepted, surrogate)
        plt.suptitle("%s, acceptance rate: %.2f" % \
                     (surrogate.__class__.__name__, np.mean(accepted)))
        
    plt.show()
