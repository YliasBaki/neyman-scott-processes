import numpy as np


def new_cluster_log_prob(point, model) -> float:
    """
    TODO
    """
    log_prob = (
        np.log(alpha) 
        + alpha * (np.log(beta) - np.log(1 + beta)) 
        + model.log_marginal_likelihood(point)
    )
    return log_prob
