import numpy as np


def gibbs_sampler(data: np.array, model, num_samples: int = 100) -> np.array:
    """
    Collapsed Gibbs sampling

    Args:
        data (Array) : The data to be clustered.
        model () : 
        num_samples (int): The number of samples to be extracted.

    """
    for i in range(num_samples):
        assignments = partition(data, model)
        # lambda_0 = simple_bkgd_intensity(assignments)


def partition(data: np.array, model, num_clusters) -> np.array:
    """
    TODO
    """
    N = data.shape[0]
    assignments = np.ones(N)*(-1)
    log_probs = np.zeros(num_clusters + 2)

    for i in range(N):
        log_probs[:num_clusters] = model.existing_cluster_log_prob(data[i])
        log_probs[-2] = model.bkgd_log_prob(data[i])
        log_probs[-1] = model.new_cluster_log_prob(data[i])

        probs = np.exp(log_probs)
        pweights = probs / np.sum(probs)
        assignments[i] = np.random.choice(num_clusters + 2, p=pweights)

    return assignments


