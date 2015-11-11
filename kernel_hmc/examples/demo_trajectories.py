from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_exp_family.examples.tools import visualise_array, pdf_grid
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian, \
    GammaEigenvalueRotatedGaussian
from kernel_hmc.proposals.kmc import KMCStatic
from kernel_hmc.tools.log import Log
import matplotlib.pyplot as plt
import numpy as np


# banana gradient depends on theano, which is an optional dependency
try:
    from kernel_hmc.densities.banana import Banana, sample_banana
    banana_available = True
except ImportError:
    banana_available = False

Log.set_loglevel(20)

def visualise_trajectory(Qs, acc_probs, log_pdf_q, target_log_pdf=None):
    assert Qs.ndim == 2
    
    if not target_log_pdf is None:
        Xs = np.linspace(-30, 30, 75)
        Ys = np.linspace(-10, 20, len(Xs))
        D, G = pdf_grid(Xs, Ys, target_log_pdf)
    
    plt.figure(figsize=(10, 12))
    plt.subplot(411)
    visualise_array(Xs, Ys, D)
    plt.plot(Qs[:, 0], Qs[:, 1])
    plt.plot(Qs[0, 0], Qs[0, 1], 'r*', markersize=15)
    plt.title("Log-pdf surrogate")
    
    plt.subplot(412)
    visualise_array(Xs, Ys, G)
    plt.plot(Qs[:, 0], Qs[:, 1])
    plt.plot(Qs[0, 0], Qs[0, 1], 'r*', markersize=15)
    plt.title("Gradient norm surrogate")
    
    plt.subplot(413)
    plt.title("Acceptance probability")
    plt.xlabel("Leap frog iteration")
    plt.plot(acc_probs)
    plt.plot([0, len(acc_probs)], [np.mean(acc_probs) for _ in range(2)], 'r--')
    plt.xlim([0, len(acc_probs)])
    
    plt.subplot(414)
    plt.title("Target log-pdf")
    plt.xlabel("Leap frog iteration")
    plt.plot(log_pdf_q)
    plt.xlim([0, len(log_pdf_q)])
    
if __name__ == '__main__':
    """
    Example that visualises trajectories of KMC lite and finite on a simple target.
    C.f. Figures 1 and 2 in the paper.
    """
    
    D = 2
    N = 1000
    
    # target is banana density, fallback to Gaussian if theano is not present
    if banana_available:
        target = Banana()
        X = sample_banana(N, D)
    else:
        target = GammaEigenvalueRotatedGaussian(gamma_shape=1., D=D)
        X = target.sample(N)
    
    # plot trajectories for both KMC lite and finite
    for surrogate in [
                        KernelExpFiniteGaussian(gamma=.5, lmbda=0.001, m=1000, D=D),
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
        current = np.array([0, -3])
        current_log_pdf = target.log_pdf(current)
        Qs, acc_probs, log_pdf_q = kmc._proposal_trajectory(current, current_log_pdf)
        
        visualise_trajectory(Qs, acc_probs, log_pdf_q, surrogate)
        plt.suptitle("%s" % surrogate.__class__.__name__)
        
    plt.show()
