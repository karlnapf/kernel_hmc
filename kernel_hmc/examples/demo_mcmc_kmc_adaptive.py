from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussianAdaptive
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian
from kernel_hmc.examples.plotting import visualise_trace
from kernel_hmc.mini_mcmc.mini_mcmc import mini_mcmc
from kernel_hmc.proposals.kmc import KMC
import matplotlib.pyplot as plt
import numpy as np
from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch


# banana gradient depends on theano, which is an optional dependency
try:
    from kernel_hmc.densities.banana import Banana
    banana_available = True
except ImportError:
    banana_available = False

if __name__ == '__main__':
    """
    This example samples from the Banana target (if theano is installed).
    It uses here uses an adaptive instance of KMC lite to start with, then (optionally) 
    switches to KMC finite using the KMC exploration as initial sketch of the target
    
    Note this is an illustrative demo and the number of iterations are set very low.
    """
    D = 2
    N = 500
    
    # target is banana density, with fallback option
    if banana_available:
        target = Banana(D=D)
    else:
        target = IsotropicZeroMeanGaussian(D=D)

    # KMC lite is geometrically ergodic on this target, use it if nothing about target is known
    # KMC finite can be used after burn-in, i.e. if some oracle samples are available
    # see below
    # this surrogate automatically learns parameters in every fit call
    surrogate = KernelExpLiteGaussianAdaptive(sigma=20., lmbda=0.001, D=D, N=N)

    # HMC parameters, step size will be adapted
    momentum = IsotropicZeroMeanGaussian(D=D)
    num_steps_min = 10
    num_steps_max = 50
    step_size_min = .1
    step_size_max = .1
    
    
    # kmc sampler instance, schedule here also controls updating the surrogate
    # this is a very liberate schedule, i.e. constant adaptation
    # necessary if KMC is not initialised with oracle samples
    schedule = lambda t: 0.001
    acc_star = 0.7
    kmc = KMC(surrogate, target,
              momentum, num_steps_min, num_steps_max, step_size_min, step_size_max,
              schedule, acc_star)
    
    # MCMC parameters
    # set to around 5000-10000 iterations to have KMC lite explored all of the support
    start = np.zeros(D)
    start[1] = -3
    num_iter = 500
    
    # run MCMC
    samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(kmc, start, num_iter, D)
    
    visualise_trace(samples, log_pdf, accepted, log_pdf_density=surrogate, step_sizes=step_sizes)
    plt.suptitle("KMC lite %s, acceptance rate: %.2f" % \
                 (surrogate.__class__.__name__, np.mean(accepted)))
    
    # now initialise KMC finite with the samples from the surrogate, and run for more
    # learn parameters before starting
    thinned = samples[np.random.permutation(len(samples))[:N]]
    surrogate2 = KernelExpFiniteGaussian(sigma=2, lmbda=0.001, D=D, m=N)
    surrogate2.set_parameters_from_dict(BayesOptSearch(surrogate2, thinned, {'sigma': [-3,3]}).optimize(3))
    surrogate2.fit(thinned)
    
    # now use conservative schedule, or None at all if confident in oracle samples
    schedule2 = lambda t: 0.01 if t < 3000 else 0.
    kmc2 = KMC(surrogate2, target,
              momentum, kmc.num_steps_min, kmc.num_steps_max, kmc.step_size[0], kmc.step_size[1],
              schedule2, acc_star)

    # run MCMC
    samples2, proposals2, accepted2, acc_prob2, log_pdf2, times2, step_sizes = mini_mcmc(kmc2, start, num_iter, D)
    visualise_trace(samples2, log_pdf2, accepted2, log_pdf_density=surrogate2, step_sizes=step_sizes)
    plt.suptitle("KMC finite, %s, acceptance rate: %.2f" % \
                 (surrogate.__class__.__name__, np.mean(accepted2)))
    plt.show()