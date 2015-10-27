from numpy.ma.testutils import assert_close

from kernel_hmc.densities.gaussian import IsotropicZeroMeanGaussian, \
    log_gaussian_pdf
import numpy as np


def test_isotropic_zero_mean_equals_log_gaussian_pdf():
    D = 2
    x = np.random.randn(D)
    g = IsotropicZeroMeanGaussian(sigma=np.sqrt(2))
    log_pdf = log_gaussian_pdf(x, mu=np.zeros(D), Sigma=np.eye(D) * 2, is_cholesky=False, compute_grad=False)
    assert_close(log_pdf, g.log_pdf(x))

def test_isotropic_zero_mean_equals_log_gaussian_pdf_grad():
    D = 2
    x = np.random.randn(D)
    g = IsotropicZeroMeanGaussian(sigma=np.sqrt(2))
    log_pdf = log_gaussian_pdf(x, mu=np.zeros(D), Sigma=np.eye(D) * 2, is_cholesky=False, compute_grad=True)
    assert_close(log_pdf, g.grad(x))

