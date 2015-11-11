# depends on shogun, which might not be available
from nose import SkipTest

import numpy as np


try:
    from kernel_hmc.densities.posterior_gp_classification_ard import GlassPosterior
    glass_available = True
except ImportError:
    glass_available = False
    


def test_glass_posterior_setup_execute():
    if not glass_available:
        raise SkipTest("Shogun not available") 
    
    GlassPosterior().set_up()

def test_glass_posterior_log_pdf_execute():
    if not glass_available:
        raise SkipTest("Shogun not available")
    
    D = 9
    theta = np.random.randn(D)
    
    target = GlassPosterior()
    target.set_up()
    
    target.log_pdf(theta)
