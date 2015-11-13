from kernel_exp_family.examples.tools import pdf_grid, visualise_array
from kernel_hmc.tools.mcmc_convergence import autocorr
import matplotlib.pyplot as plt
import numpy as np

def visualise_trajectory(Qs, acc_probs, log_pdf_q, D, log_pdf=None):
    assert Qs.ndim == 2
    
    plot_density = log_pdf is not None and D==2
    
    plt.figure(figsize=(10, 12))
    plt.subplot(411)
    
    # plot density if given and dimension is 2
    if plot_density:
        Xs = np.linspace(-30, 30, 75)
        Ys = np.linspace(-10, 20, len(Xs))
        D, G = pdf_grid(Xs, Ys, log_pdf)
        visualise_array(Xs, Ys, D)
    
    plt.plot(Qs[:, 0], Qs[:, 1])
    plt.plot(Qs[0, 0], Qs[0, 1], 'r*', markersize=15)
    plt.title("Log-pdf surrogate")
    
    plt.subplot(412)
    if plot_density:
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

def visualise_trace(samples, log_pdf_trajectory, accepted, log_pdf_density=None, idx0=0, idx1=1):
    assert samples.ndim == 2
    
    D = samples.shape[1]
    
    plt.figure(figsize=(15, 12))
    
    plt.subplot(421)
    plt.plot(samples[:, idx0])
    plt.title("Trace $x_%d$" % (idx0+1))
    plt.xlabel("MCMC iteration")
    plt.grid(True)
    
    plt.subplot(422)
    plt.plot(samples[:, idx1])
    plt.title("Trace $x_%d$" % (idx1+1))
    plt.xlabel("MCMC iteration")
    plt.grid(True)
    
    plt.subplot(423)
    if not log_pdf_density is None and D == 2:
        Xs = np.linspace(-28, 28, 50)
        Ys = np.linspace(-6, 16, len(Xs))
        D, _ = pdf_grid(Xs, Ys, log_pdf_density)
        visualise_array(Xs, Ys, D)
        
    plt.plot(samples[:, idx0], samples[:, idx1])
    plt.title("Trace $(x_%d, x_%d)$" % (idx0+1, idx1+1))
    plt.grid(True)
    plt.xlabel("$x_%d$" % (idx0+1))
    plt.ylabel("$x_%d$" % (idx1+1))
    
    plt.subplot(424)
    plt.plot(log_pdf_trajectory)
    plt.title("log pdf along trajectory")
    plt.xlabel("MCMC iteration")
    plt.grid(True)
    
    plt.subplot(425)
    plt.plot(autocorr(samples[:, idx0]))
    plt.title("Autocorrelation $x_%d$" % (idx0+1))
    plt.xlabel("Lag")
    plt.grid(True)
    
    plt.subplot(426)
    plt.plot(autocorr(samples[:, idx1]))
    plt.title("Autocorrelation $x_%d$" % (idx1+1))
    plt.xlabel("Lag")
    plt.grid(True)
    
    plt.subplot(427)
    plt.plot(np.cumsum(accepted) / np.arange(1, len(accepted)+1))
    plt.title("Average acceptance rate")
    plt.xlabel("MCMC iterations")
    plt.grid(True)
