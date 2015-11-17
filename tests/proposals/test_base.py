
from nose.tools import assert_raises

from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian
from kernel_hmc.proposals.hmc import HMCBase
from kernel_hmc.proposals.kmc import KMCStatic
from kernel_hmc.tools.assertions import assert_array_shape
import numpy as np
from tests.proposals.test_kmc import get_static_surrogate


def get_hmc_kernel():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, momentum = get_target_momentum()
    hmc = HMCBase(target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)
    return hmc

def get_kmc_static_kernel():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, momentum = get_target_momentum()

    N = 200
    X = np.random.randn(N, momentum.D)
    est = KernelExpLiteGaussian(sigma=1, lmbda=.1, D=momentum.D, N=N)
    est.fit(X)
    
    surrogate = get_static_surrogate(momentum.D)
    kmc = KMCStatic(surrogate, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)
    
    return kmc

def get_all_kernels():
    
    
    proposals = [
                 get_hmc_kernel(),
                 get_kmc_static_kernel(),
                 ]
    
    return proposals

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

def test_hmc_base_init_execute():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, momentum = get_target_momentum()
    HMCBase(target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)

def test_hmc_base_init_wrong_target_type():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    _, momentum = get_target_momentum()
    target = None
    assert_raises(ValueError, HMCBase, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)

def test_hmc_base_init_wrong_target_logpdf():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, momentum = get_target_momentum()
    target.log_pdf = None
    assert_raises(ValueError, HMCBase, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)

def test_hmc_base_init_wrong_target_grad():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, momentum = get_target_momentum()
    target.grad = None
    assert_raises(ValueError, HMCBase, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)

def test_hmc_base_init_wrong_momentum_type():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, _ = get_target_momentum()
    momentum = None
    assert_raises(TypeError, HMCBase, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)

def test_hmc_base_init_wrong_momentum_log_pdf():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, momentum = get_target_momentum()
    momentum.log_pdf = None
    assert_raises(ValueError, HMCBase, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)

def test_hmc_base_init_wrong_momentum_grad():
    num_steps_min, num_steps_max, step_size_min, step_size_max = get_hmc_parameters()
    target, momentum = get_target_momentum()
    momentum.grad = None
    assert_raises(ValueError, HMCBase, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)


def test_all_proposal_execute_result():
    for kernel in get_all_kernels():
        current = np.zeros(kernel.momentum.D)
        current_log_pdf = 0.
        q, acc_prob, log_pdf_q = kernel.proposal(current, current_log_pdf)
        
        assert_array_shape(q, ndim=1, shape=(kernel.momentum.D,))
        
        assert type(log_pdf_q) is np.float64
        assert type(acc_prob) is np.float64
        
        assert acc_prob >= 0 and acc_prob <= 1.

def test_all_accept_prob_log_pdf_execute_result():
    for kernel in get_all_kernels():
        
        current = np.zeros(kernel.momentum.D)
        q = current.copy()
        current_log_pdf = 0.
        p0_log_pdf = 0.
        p_log_pdf = 0.
        
        acc_prob, log_pdf_q = kernel.accept_prob_log_pdf(current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        assert type(log_pdf_q) is np.float64
        assert type(acc_prob) is np.float64
        
        assert acc_prob >= 0 and acc_prob <= 1.
