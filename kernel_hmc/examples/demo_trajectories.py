from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian,\
    sample_gaussian
from kernel_hmc.examples.plotting import visualise_trajectory
from kernel_hmc.proposals.kmc import KMCStatic
import matplotlib.pyplot as plt


# banana gradient depends on theano, which is an optional dependency
try:
    from kernel_hmc.densities.banana import Banana, sample_banana
    banana_available = True
except ImportError:
    banana_available = False

if __name__ == '__main__':
    """
    Example that visualises trajectories of KMC lite and finite on a simple target.
    C.f. Figures 1 and 2 in the paper.
    """
    
    # for D=2, the fitted log-density is plotted, otherwise trajectory only
    D = 2
    N = 1000
    
    # target is banana density, fallback to Gaussian if theano is not present
    if banana_available:
        target = Banana()
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
        
        
        # HMC parameters
        momentum = IsotropicZeroMeanGaussian(D=D, sigma=.1)
        num_steps = 1000
        step_size = .01
        
        # kmc sampler instance
        kmc = KMCStatic(surrogate, target, momentum, num_steps, num_steps, step_size, step_size)
        
        # simulate trajectory from starting point, note _proposal_trajectory is a "hidden" method
        current = X[0]
        current_log_pdf = target.log_pdf(current)
        Qs, acc_probs, log_pdf_q = kmc._proposal_trajectory(current, current_log_pdf)
        
        visualise_trajectory(Qs, acc_probs, log_pdf_q, D, surrogate)
        plt.suptitle("%s" % surrogate.__class__.__name__)
        
    plt.show()
