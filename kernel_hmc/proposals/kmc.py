from kernel_hmc.proposals.base import HMCBase
from kernel_hmc.tools.assertions import assert_implements_log_pdf_and_grad

class KMCStatic(HMCBase):
    """
    """
    
    def __init__(self, surrogate, target, momentum, num_steps_min=10, num_steps_max=100, step_size_min=0.05,
                 step_size_max=0.3):
        """
        """
        HMCBase.__init__(self, target, momentum, num_steps_min, num_steps_max, step_size_min, step_size_max)
        assert_implements_log_pdf_and_grad(surrogate)
        
        self.target = surrogate
        self.orig_target = target
    
    def accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf):
        # same as super-class, but with original target
        kernel_target = self.target
        self.target = self.orig_target
        
        acc_prob, log_pdf_q = HMCBase.accept_prob_log_pdf(self, current, q, p0_log_pdf, p_log_pdf, current_log_pdf)
        
        # restore target
        self.target = kernel_target
        
        return acc_prob, log_pdf_q