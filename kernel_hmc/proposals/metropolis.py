from kernel_hmc.densities.gaussian import sample_gaussian
from kernel_hmc.proposals.base import ProposalBase, standard_sqrt_schedule
from kernel_hmc.tools.assertions import assert_implements_log_pdf_and_grad
from kernel_hmc.tools.log import Log
import numpy as np

logger = Log.get_logger()

# low rank update depends on "cholupdate" optional dependency
try:
    from choldate._choldate import cholupdate
    cholupdate_available = True
except ImportError:
    cholupdate_available = False
    logger.warning("Package cholupdate not available. Adaptive Metropolis falls back to (more expensive) re-estimation of covariance.")

if cholupdate_available:
    def rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda=.1, mean=None, cov_L=None, nu2=1., gamma2=None):
        """
        Returns updated mean and Cholesky of sum of outer products following a
        (1-lmbda)*old + lmbda* step_size*uu^T+lmbda*gamm2*I
        rule
        
        Optional: If gamma2 is given, an isotropic term gamma2 * I is added to the uu^T part
        
        where old mean and cov_L=Cholesky(old) (lower Cholesky) are given.
        
        Performs efficient rank-one updates of the Cholesky directly.
        """
        assert lmbda >= 0 and lmbda <= 1
        assert u.ndim == 1
        D = len(u)
        
        # check if first term
        if mean is None or cov_L is None :
            # in that case, zero mean and scaled identity matrix
            mean = np.zeros(D)
            cov_L = np.eye(D) * nu2
        else:
            assert len(mean) == D
            assert mean.ndim == 1
            assert cov_L.ndim == 2
            assert cov_L.shape[0] == D
            assert cov_L.shape[1] == D
        
        # update mean
        updated_mean = (1 - lmbda) * mean + lmbda * u
        
        # update Cholesky: first downscale existing Cholesky
        update_cov_L = np.sqrt(1 - lmbda) * cov_L.T
        
        # rank-one update of the centered new vector
        update_vec = np.sqrt(lmbda) * np.sqrt(nu2) * (u - mean)
        cholupdate(update_cov_L, update_vec)
        
        # optional: add isotropic term if specified, requires looping rank-one updates over
        # all basis vectors e_1, ..., e_D
        if gamma2 is not None:
            e_d = np.zeros(D)
            for d in range(D):
                e_d[:] = 0
                e_d[d] = np.sqrt(gamma2)
                
                # could do a Cholesky update, but this routine does a loop over dimensions
                # where the vector only has one non-zero component
                # That is O(D^2) and therefore not efficient when used in a loop
                cholupdate(update_cov_L, np.sqrt(lmbda) * e_d)
                
                # TODO:
                # in contrast, can do a simplified update when knowing that e_d is sparse
                # manual Cholesky update (only doing the d-th component of algorithm on
                # https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
    #             # wiki (MB) code:
    #             r = sqrt(L(k,k)^2 + x(k)^2);
    #             c = r / L(k, k);
    #             s = x(k) / L(k, k);
    #             L(k, k) = r;
    #             L(k+1:n,k) = (L(k+1:n,k) + s*x(k+1:n)) / c;
    #             x(k+1:n) = c*x(k+1:n) - s*L(k+1:n,k);
    
        # since cholupdate works on transposed version
        update_cov_L = update_cov_L.T
        
        # done updating Cholesky
        
        return updated_mean, update_cov_L

def rank_update_mean_covariance_cholesky_lmbda_naive(u, lmbda=.1, mean=None, cov_L=None, nu2=1., gamma2=None):
    """
    Returns updated mean and Cholesky of sum of outer products following a
    (1-lmbda)*old + lmbda* step_size*uu^T
    rule
    
    Optional: If gamma2 is given, an isotropic term gamma2 * I is added to the uu^T part
    
    where old mean and cov_L=Cholesky(old) (lower Cholesky) are given.
    
    Naive version that re-computes the Cholesky factorisation
    """
    assert lmbda >= 0 and lmbda <= 1
    assert u.ndim == 1
    D = len(u)
    
    # check if first term
    if mean is None or cov_L is None :
        # in that case, zero mean and scaled identity matrix
        mean = np.zeros(D)
        cov_L = np.eye(D) * nu2
    else:
        assert len(mean) == D
        assert mean.ndim == 1
        assert cov_L.ndim == 2
        assert cov_L.shape[0] == D
        assert cov_L.shape[1] == D
    
    # update mean
    updated_mean = (1 - lmbda) * mean + lmbda * u
    
    # centered new vector
    update_vec = u - mean

    # reconstruct covariance, update
    update_cov = np.dot(cov_L, cov_L.T)
    update_cov = (1 - lmbda)*update_cov + lmbda*nu2*np.outer(update_vec, update_vec)
    
    # optional: add isotropic term if specified
    if gamma2 is not None:
        update_cov += np.eye(update_cov.shape[0])*gamma2
    
    # re-compute Cholesky
    update_cov_L = np.linalg.cholesky(update_cov)
    
    return updated_mean, update_cov_L


class AdaptiveMetropolis(ProposalBase):
    """
    Implements the adaptive MH.
    
    If "cholupdate" package is available, 
    performs efficient low-rank updates of Cholesky factor of covariance,
    costing O(d^2) computation.
    
    Otherwise, covariance is is simply updated every iteration and its Cholesky
    factorisation is re-computed every time, costing O(d^3) computation.
    """
    
    def __init__(self, target, D, step_size, gamma2,
                 adaptation_schedule=standard_sqrt_schedule, acc_star=0.234):
        """
        target        - Target density, must implement log_pdf method
        D             - Target dimension
        step_size           - Scaling parameter for covariance
        gamma2        - Exploration parameter. Added to learned variance
        adaptation_schedule      - Optional. Function that generates adaptation weights
                        given the MCMC iteration number.
                        The weights are used in the stochastic updating of the
                        covariance.
                        
                        If not set, internal covariance is never updated. In that case, call
                        batch_covariance() before using.
        acc_star        Optional: If set, the step_size parameter is tuned so that
                        average acceptance equals acc_star, using the same adaptation_schedule
                        as for the chain history update (If adaptation_schedule is set, otherwise
                        ignored)
        """
        ProposalBase.__init__(self, D, step_size, adaptation_schedule, acc_star)
        
        assert_implements_log_pdf_and_grad(target, assert_grad=False)
        
        self.target = target
        self.D = D
        self.gamma2 = gamma2
        
        self._initialise()
    
    def _initialise(self):
        """
        Initialises internal state. To be called before MCMC chain starts.
        """
        if self.adaptation_schedule is not None:
            # start from scratch
            self.mu = np.zeros(self.D)
            
            # initialise as scaled isotropic, otherwise Cholesky updates fail
            self.L_C = np.eye(self.D) * np.sqrt(self.step_size)
        else:
            # make user call the set_batch_covariance() function
            self.mu = None
            self.L_C = None

    def set_batch_covariance(self, Z):
        self.mu = np.mean(Z, axis=0)
        self.L_C = np.linalg.cholesky(self.step_size*np.cov(Z.T)+np.eye(Z.shape[1])*self.gamma2)
    
    def update(self, samples, acc_probs):
        self.t += 1
        
        z_new = samples[-1]
        previous_accpept_prob = acc_probs[-1]
        if self.adaptation_schedule is not None:
            # generate updating weight
            lmbda = self.adaptation_schedule(self.t)
            
            logger.debug("Updating covariance using lmbda=%.3f" % lmbda)
            if cholupdate_available:
                # low-rank update of Cholesky, costs O(d^2) only, adding exploration noise on the fly
                self.mu, self.L_C = rank_one_update_mean_covariance_cholesky_lmbda(z_new,
                                                                                   lmbda,
                                                                                   self.mu,
                                                                                   self.L_C,
                                                                                   self.step_size,
                                                                                   self.gamma2)
            else:
                # low-rank update of Cholesky, naive costs O(d^3), adding exploration noise on the fly
                self.mu, self.L_C = rank_update_mean_covariance_cholesky_lmbda_naive(z_new,
                                                                                   lmbda,
                                                                                   self.mu,
                                                                                   self.L_C,
                                                                                   self.step_size,
                                                                                   self.gamma2)
            
            # update scalling parameter if wanted
            if self.acc_star is not None:
                self._update_scaling(lmbda, previous_accpept_prob)
    
    def proposal(self, current, current_log_pdf):
        """
        Returns a sample from the proposal centred at current, acceptance probability,
        and its log-pdf under the target.
        """
        if self.adaptation_schedule is None and (self.mu is None or self.L_C is None):
            raise ValueError("AM has not seen data yet." \
                             "Either call set_batch_covariance() or set update schedule")
        
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)

        # generate proposal
        proposal = sample_gaussian(N=1, mu=current, Sigma=self.L_C, is_cholesky=True)[0]
        proposal_log_pdf = self.target.log_pdf(proposal)
        
        # compute acceptance prob, proposals probability cancels due to symmetry
        acc_log_prob = np.min([0, proposal_log_pdf - current_log_pdf])
        
        # probability of proposing current when would be sitting at proposal is symmetric
        return proposal, np.exp(acc_log_prob), proposal_log_pdf
    

class StandardMetropolis(AdaptiveMetropolis):
    """
    Implements the adaptive MH with a isotropic proposal covariance.
    """
    
    def __init__(self, target, D, step_size, gamma2,
                 adaptation_schedule=standard_sqrt_schedule, acc_star=0.234):
        """
        target        - Target density, must implement log_pdf method
        D             - Target dimension
        step_size           - Scaling parameter for covariance
        gamma2        - Exploration parameter. Added to learned variance
        adaptation_schedule      - Optional. Function that generates adaptation weights
                        given the MCMC iteration number.
                        The weights are used in the stochastic updating of the
                        covariance.
                        
                        If not set, internal covariance is never updated.
        acc_star        Optional: If set, the step_size parameter is tuned so that
                        average acceptance equals acc_star, using the same adaptation_schedule
                        as for the chain history update (If adaptation_schedule is set, otherwise
                        ignored)
        """
        AdaptiveMetropolis.__init__(self, target, D, step_size, gamma2,
                                    adaptation_schedule, acc_star)
        
        assert_implements_log_pdf_and_grad(target, assert_grad=False)
        
        self.target = target
        self.D = D
        self.gamma2 = gamma2

    def proposal(self, current, current_log_pdf):
        """
        Returns a sample from the proposal centred at current, acceptance probability,
        and its log-pdf under the target.
        """
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)

        # generate proposal
        proposal = sample_gaussian(N=1, mu=current, Sigma=np.eye(self.D) * np.sqrt(self.step_size), is_cholesky=True)[0]
        proposal_log_pdf = self.target.log_pdf(proposal)
        
        # compute acceptance prob, proposals probability cancels due to symmetry
        acc_log_prob = np.min([0, proposal_log_pdf - current_log_pdf])
        
        # probability of proposing current when would be sitting at proposal is symmetric
        return proposal, np.exp(acc_log_prob), proposal_log_pdf
    
