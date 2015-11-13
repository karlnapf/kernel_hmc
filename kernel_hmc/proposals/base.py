from abc import abstractmethod

from kernel_hmc.densities.gaussian import GaussianBase
from kernel_hmc.hamiltonian.leapfrog import leapfrog_no_storing, leapfrog
from kernel_hmc.tools.assertions import assert_positive_int,\
    assert_implements_log_pdf_and_grad, assert_positive_float,\
    assert_inout_log_pdf_and_grad
from kernel_hmc.tools.log import logger
import numpy as np

def standard_sqrt_schedule(t):
    return 1. / np.sqrt(t + 1)

class ProposalBase(object):
    def __init__(self, D, step_size, adaptation_schedule, acc_star):
        self.D = D
        self.step_size = step_size
        self.adaptation_schedule = adaptation_schedule
        self.acc_star = acc_star
        
        self.t = 0
        
        # some sanity checks
        assert acc_star is None or acc_star > 0 and acc_star < 1
        if adaptation_schedule is not None:
            lmbdas = np.array([adaptation_schedule(t) for t in  np.arange(100)])
            assert np.all(lmbdas >= 0)
            assert np.allclose(np.sort(lmbdas)[::-1], lmbdas)
    
    def initialise(self):
        pass
    
    def proposal(self):
        pass
    
    def update(self, samples, acc_probs):
        self.t += 1
        
        previous_accpept_prob = acc_probs[-1]
        
        if self.adaptation_schedule is not None:
            # generate updating weight
            lmbda = self.adaptation_schedule(self.t)
            
            if np.random.rand() <= lmbda:
                if self.acc_star is not None:
                    self._update_scaling(lmbda, previous_accpept_prob)
    
    def _update_scaling(self, lmbda, accept_prob):
        # difference desired and actuall acceptance rate
        diff = accept_prob - self.acc_star
        
        new_log_step_size = np.log(self.step_size) + lmbda * diff
        new_step_size = np.exp(new_log_step_size)
        
        logger.info("Acc. prob. diff. was %.3f-%.3f=%.3f. Updating step-size from %s to %s." % \
                     (accept_prob, self.acc_star, diff, self.step_size, new_step_size))

        self.new_step_size = new_step_size

class HMCBase(ProposalBase):
    def __init__(self, target, momentum, num_steps_min=10, num_steps_max=100, step_size_min=0.05,
                 step_size_max=0.3, adaptation_schedule=standard_sqrt_schedule, acc_star=0.7):
        
        if not isinstance(momentum, GaussianBase):
            raise TypeError("Momentum (%s) must be subclass of %s" % \
                            (str(type(momentum)), str(GaussianBase)))
        assert_implements_log_pdf_and_grad(target)
        assert_implements_log_pdf_and_grad(momentum)
        assert_inout_log_pdf_and_grad(target, momentum.D, assert_grad=False)
            
        assert_positive_int(num_steps_min)
        assert_positive_int(num_steps_max)
        if not num_steps_min<=num_steps_max:
            raise ValueError("Minimum number of leapfrog steps (%d) must be larger than maximum number (%d)." % \
                             (num_steps_min, num_steps_max))
        
        assert_positive_float(step_size_min)
        assert_positive_float(step_size_max)
        if not num_steps_min<=num_steps_max:
            raise ValueError("Minimum size of leapfrog steps (%d) must be larger than maximum size (%d)." % \
                             (step_size_min, step_size_max))
        
        step_size = np.array([step_size_min, step_size_max])
        ProposalBase.__init__(self, momentum.D, step_size, adaptation_schedule, acc_star)
        
        self.target = target
        self.momentum = momentum
        self.num_steps_min = num_steps_min
        self.num_steps_max = num_steps_max
        
    
    def _proposal_trajectory(self, current, current_log_pdf):
        # sample momentum and leapfrog parameters
        p0 = self.momentum.sample()
        p0_log_pdf = self.momentum.log_pdf(p0)
        num_steps = np.random.randint(self.num_steps_min, self.num_steps_max + 1)
        step_size = np.random.rand() * (self.step_size[1] - self.step_size[0]) + self.step_size[0]
        
        logger.debug("Simulating Hamiltonian flow trajectory.")
        Qs, Ps = leapfrog(current, self.target.grad, p0, self.momentum.grad, step_size, num_steps)
        
        # compute acceptance probability, extracting log_pdf of q
        logger.debug("Computing acceptance probabilies.")
        acc_probs = np.zeros(len(Qs))
        log_pdf_q = np.zeros(len(Qs))
        
        for i in range(len(Qs)):
            p = Ps[i]
            q = Qs[i]
            p_log_pdf = self.momentum.log_pdf(p)
            acc_probs[i], log_pdf_q[i] = self.accept_prob_log_pdf(current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        return Qs, acc_probs, log_pdf_q
    
    def proposal(self, current, current_log_pdf):
        """
        """
        
        # sample momentum and leapfrog parameters
        p0 = self.momentum.sample()
        p0_log_pdf = self.momentum.log_pdf(p0)
        num_steps = np.random.randint(self.num_steps_min, self.num_steps_max + 1)
        step_size = np.random.rand() * (self.step_size[1] - self.step_size[0]) + self.step_size[0]
        
        logger.debug("Simulating Hamiltonian flow.")
        q, p = leapfrog_no_storing(current, self.target.grad, p0, self.momentum.grad, step_size, num_steps)
        
        # compute acceptance probability, extracting log_pdf of q
        logger.debug("Computing acceptance probability.")
        p_log_pdf = self.momentum.log_pdf(p)
        acc_prob, log_pdf_q = self.accept_prob_log_pdf(current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        return q, acc_prob, log_pdf_q
    
    @abstractmethod
    def accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf=None):
        # potentially re-use log_pdf of last accepted state
        if current_log_pdf is None:
            current_log_pdf = self.target.log_pdf(current)
        
        log_pdf_q = self.target.log_pdf(q)
        H0 = -current_log_pdf - p0_log_pdf
        H = -log_pdf_q - p_log_pdf
        difference = -H + H0
        acc_prob = np.exp(np.minimum(0., difference))
        
        return acc_prob, log_pdf_q
