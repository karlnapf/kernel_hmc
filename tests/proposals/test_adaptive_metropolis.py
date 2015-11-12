from nose import SkipTest
from numpy.testing.utils import assert_allclose

from kernel_hmc.proposals.metropolis import rank_update_mean_covariance_cholesky_lmbda_naive
import numpy as np


# low rank update depends on "cholupdate" optional dependency
try:
    from kernel_hmc.proposals.metropolis import rank_one_update_mean_covariance_cholesky_lmbda
    low_rank_update_available = True
except ImportError:
    low_rank_update_available = False
    

def test_rank_update_mean_covariance_cholesky_lmbda_cholupdate_equals_naive():
    if not low_rank_update_available:
        raise SkipTest("cholupdate not available")
    
    D = 3
    N = 100
    X = np.random.randn(N, D)
    
    mean = np.mean(X, 0)
    Sigma = np.cov(X.T)
    L = np.linalg.cholesky(Sigma)
    assert_allclose(np.dot(L, L.T), Sigma)
    
    # update with one more vector
    u = np.random.randn(D)
    lmbda = 0.1
    nu2 = 0.2
    gamma2 = 0.3
    
    m_test_naive, L_test_naive = rank_update_mean_covariance_cholesky_lmbda_naive(u, lmbda, mean, L, nu2=nu2, gamma2=gamma2)
    m_test, L_test = rank_update_mean_covariance_cholesky_lmbda_naive(u, lmbda, mean, L, nu2=nu2, gamma2=gamma2)
    
    assert_allclose(m_test_naive, m_test)
    assert_allclose(L_test_naive, L_test)

def test_rank_update_mean_covariance_cholesky_lmbda_naive():
    D = 3
    N = 100
    X = np.random.randn(N, D)
    
    mean = np.mean(X, 0)
    Sigma = np.cov(X.T)
    L = np.linalg.cholesky(Sigma)
    assert_allclose(np.dot(L, L.T), Sigma)
    
    # update with one more vector
    u = np.random.randn(D)
    lmbda = 0.1
    nu2 = 0.2
    gamma2 = 0.3
    
    updated_mean = (1 - lmbda) * mean + lmbda * u
    updated_Sigma = (1 - lmbda) * Sigma + lmbda * nu2 * np.outer(u - mean, u - mean) + np.eye(D) * gamma2
    updated_L = np.linalg.cholesky(updated_Sigma)
    
    m_test, L_test = rank_update_mean_covariance_cholesky_lmbda_naive(u, lmbda, mean, L, nu2=nu2, gamma2=gamma2)
    assert_allclose(updated_mean, m_test)
    assert_allclose(updated_L, L_test)

def test_rank_one_update_mean_covariance_cholesky_lmbda():
    if not low_rank_update_available:
        raise SkipTest("cholupdate not available")
    
    D = 3
    N = 100
    X = np.random.randn(N, D)
    
    mean = np.mean(X, 0)
    Sigma = np.cov(X.T)
    L = np.linalg.cholesky(Sigma)
    assert_allclose(np.dot(L, L.T), Sigma)
    
    # update with one more vector
    u = np.random.randn(D)
    lmbda = 0.1
    
    updated_mean = (1 - lmbda) * mean + lmbda * u
    updated_Sigma = (1 - lmbda) * Sigma + lmbda * np.outer(u - mean, u - mean)
    updated_L = np.linalg.cholesky(updated_Sigma)
    
    m_test, L_test = rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda, mean, L)
    assert_allclose(updated_mean, m_test)
    assert_allclose(updated_L, L_test)

def test_rank_one_update_mean_covariance_cholesky_lmbda_gamma2():
    if not low_rank_update_available:
        raise SkipTest("cholupdate not available")
    
    D = 3
    N = 100
    X = np.random.randn(N, D)
    gamma2 = 2.
    
    mean = np.mean(X, 0)
    Sigma = np.cov(X.T)
    L = np.linalg.cholesky(Sigma)
    assert_allclose(np.dot(L, L.T), Sigma)
    
    # update with one more vector
    u = np.random.randn(D)
    lmbda = 0.1
    
    updated_mean = (1 - lmbda) * mean + lmbda * u
    updated_Sigma = (1 - lmbda) * Sigma + lmbda * np.outer(u - mean, u - mean) + lmbda * gamma2 * np.eye(D)
    updated_L = np.linalg.cholesky(updated_Sigma)
    
    m_test, L_test = rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda, mean, L, gamma2=gamma2)
    assert_allclose(updated_mean, m_test)
    assert_allclose(updated_L, L_test)

def test_rank_one_update_mean_covariance_cholesky_lmbda_nu2():
    if not low_rank_update_available:
        raise SkipTest("cholupdate not available")
    
    D = 3
    N = 100
    X = np.random.randn(N, D)
    nu2 = 2.
    
    mean = np.mean(X, 0)
    Sigma = np.cov(X.T)
    L = np.linalg.cholesky(Sigma)
    assert_allclose(np.dot(L, L.T), Sigma)
    
    # update with one more vector
    u = np.random.randn(D)
    lmbda = 0.1
    
    updated_mean = (1 - lmbda) * mean + lmbda * u
    updated_Sigma = (1 - lmbda) * Sigma + lmbda * nu2 * np.outer(u - mean, u - mean)
    updated_L = np.linalg.cholesky(updated_Sigma)
    
    m_test, L_test = rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda, mean, L, nu2=2.)
    assert_allclose(updated_mean, m_test)
    assert_allclose(updated_L, L_test)
