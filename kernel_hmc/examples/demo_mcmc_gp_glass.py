from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussianAdaptive
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian
from kernel_hmc.examples.demo_mcmc_kmc_static import visualise_trace
from kernel_hmc.mini_mcmc.mini_mcmc import mini_mcmc
from kernel_hmc.proposals.base import standard_sqrt_schedule
from kernel_hmc.proposals.kmc import KMC
from kernel_hmc.proposals.metropolis import AdaptiveMetropolis,\
    StandardMetropolis, KernelAdaptiveMetropolis
from kernel_hmc.tools.log import Log
import matplotlib.pyplot as plt
import numpy as np


logger = Log.get_logger()

# depends on optional dependency shogun
try:
    from kernel_hmc.densities.posterior_gp_classification_ard import GlassPosterior
    glass_available = True
except ImportError:
    logger.warning("Optional dependency Shogun not available, using Gaussian instead.")
    glass_available = False

def get_am_instance(target):
    # adaptive version that tunes itself towards the "optimal" acceptance rate
    # set schedule=None for completely non-adaptive version
    step_size = 1.
    gamma2 = 0.1
    schedule = standard_sqrt_schedule
    acc_star = 0.234
    am = AdaptiveMetropolis(target, D, step_size, gamma2, schedule, acc_star)
    
    return am

def get_mh_instance(target):
    # adaptive version that tunes itself towards the "optimal" acceptance rate
    step_size = 0.7
    schedule = standard_sqrt_schedule
    acc_star = 0.234
    mh = StandardMetropolis(target, D, step_size, schedule, acc_star)
    
    return mh

def get_kam_instance(target):
    # adaptive version that tunes itself towards the "optimal" acceptance rate
    step_size = 1.
    gamma2 = 0.1
    schedule = standard_sqrt_schedule
    acc_star = 0.234
    kernel_sigma = 1.
    
    return KernelAdaptiveMetropolis(target, D, N=200, kernel_sigma=kernel_sigma,
                                    gamma2=gamma2, step_size=step_size,
                                    adaptation_schedule=schedule, acc_star=acc_star)

def get_kmc_instance(target):
    step_size_min = 0.01
    step_size_max = 0.1
    num_steps_min = 1
    num_steps_max = 10
    momentum = IsotropicZeroMeanGaussian(D=D)
    schedule = standard_sqrt_schedule
    acc_star = 0.7
    
    # fully automatic parameter tuning in every fit call
    surrogate = KernelExpLiteGaussianAdaptive(sigma=1., lmbda=.001, D=D, N=200,
                                        n_initial=3, n_iter=3, minimum_size_learning=100,
                                        n_initial_relearn=3, n_iter_relearn=3,
                                        param_bounds={'sigma': [-3, 3]}
                                        )
    
    return KMC(surrogate, target,
               momentum, num_steps_min, num_steps_max, step_size_min, step_size_max,
               adaptation_schedule=schedule, acc_star=acc_star)

if __name__ == '__main__':
    """
    This example samples from the marginal posterior over hyper-parameters of a
    Gaussian Process classification model.
    
    All samplers in the paper are used.
    
    Note this is an illustrative demo and the number of iterations are set very low.
    """
    
    # Glass posterior has 9 dimensions
    D = 9
    if glass_available:
        target = GlassPosterior()
        target.set_up()
    else:
        target = IsotropicZeroMeanGaussian(D=D)

    # transition kernel, pick any
    samplers = [
                get_am_instance(target),
                get_mh_instance(target),
                get_kam_instance(target),
                get_kmc_instance(target)
               ]
    
    for sampler in samplers:
        
        # MCMC parameters
        # small number of iterations here to keep runtime short, feel free to increase
        start = np.zeros(D)
        num_iter = 50
        
        # run MCMC
        samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(sampler, start, num_iter, D)
        
        visualise_trace(samples, log_pdf, accepted, step_sizes, idx0=1, idx1=6)
        
        plt.suptitle("%s on %s, acceptance rate: %.2f" % \
                     (sampler.__class__.__name__, target.__class__.__name__, np.mean(accepted)))
        
    plt.show()
