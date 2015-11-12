from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian
from kernel_hmc.examples.demo_mcmc_kmc_static import visualise_trace
from kernel_hmc.mini_mcmc.mini_mcmc import mini_mcmc
from kernel_hmc.proposals.adaptive_metropolis import AdaptiveMetropolis
from kernel_hmc.tools.log import logger
import matplotlib.pyplot as plt
import numpy as np

# depends on optional dependency shogun
try:
    from kernel_hmc.densities.posterior_gp_classification_ard import GlassPosterior
    glass_available = True
except ImportError:
    logger.warning("Optional dependency Shogun not available, using Gaussian instead.")
    glass_available = False

def get_am_instance(target):
    # adaptive version that tunes itself towards the "optimal" acceptance rate
    nu2 = 1.
    gamma2 = 0.1
    schedule = lambda t: 1. / (t + 1) ** 0.5
    acc_star = 0.234
    am = AdaptiveMetropolis(target, D, nu2, gamma2, schedule, acc_star)
    
    return am

if __name__ == '__main__':
    # Glass posterior has 9 dimensions
    D = 9
    if glass_available:
        target = GlassPosterior()
        target.set_up()
    else:
        target = IsotropicZeroMeanGaussian(D=D)

    # transition kernel, pick any ob the above
    sampler = get_am_instance(target)
    
    # MCMC parameters
    # small number of iterations here to keep runtime short, feel free to increase
    start = np.zeros(D)
    num_iter = 50
    
    # run MCMC
    samples, proposals, accepted, acc_prob, log_pdf, times = mini_mcmc(sampler, start, num_iter, D)
    
    visualise_trace(samples, log_pdf, accepted, idx0=1, idx1=6)
    
    plt.suptitle("%s on %s, acceptance rate: %.2f" % \
                 (sampler.__class__.__name__, target.__class__.__name__, np.mean(accepted)))
    
    plt.show()
