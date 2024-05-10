import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def monte_carlo_integration(
    function, domain, N, **kwargs
) -> float:
    """
    TODO
    """
    domain = np.array(domain)
    x_list = np.random.default_rng().uniform(domain[0], domain[1], [N, 2])
    f = function(x_list, **kwargs)
    integral_estim = f.sum() * volume(domain) / f.shape[0]

    return integral_estim

def volume(domain):
    """
    TODO
    """
    domain = np.array(domain)
    delta_domain = domain[1] - domain[0]
    return np.prod(delta_domain)

