import numpy as np
import util
import scipy.stats as st
from scipy.special import gamma


class NeymanScottModel:
    """
    TODO
    """
    def __init__(
            self, domain_measure: float, alpha: float, beta: float
    ) -> None:
        """
        TODO
        """
        self.domain_measure = domain_measure
        self.alpha = alpha
        self.beta = beta

    def new_cluster_log_prob(self, x) -> float:
        """
        TODO
        """
        return (
            np.log(self.alpha)
            + np.log(self.domain_measure)
            + self.alpha * (np.log(self.beta) - np.log(1 + self.beta))
            + self.log_evidence(x)
        )

    def existing_cluster_log_prob(self, x, cluster_size) -> float:
        """
        TODO
        """
        return (
            np.log(cluster_size + self.alpha)
            * self.log_predictive(x, cluster)
        )

    def bkgd_log_prob(self, x) -> float:
        """
        TODO
        """
        pass

    def log_evidence(self, x):
        pass

    def log_predictive(self, x, cluster):
        pass


class GaussianModel(NeymanScottModel):
    """
    TODO
    """
    def __init__(
        self,
        cluster_rate: float, bounds: float,
        alpha: float, beta: float,
        cov_df: int, cov_scale: float
    ) -> None:
        """
        Args:
            cluster_rate (float): 
            bounds (float): The bounds of the (retangular) domain.
            alpha (float): The shape of the gamma distribution.
            beta (float): The rate of the gamma distribution.
            cov_df (int): Degrees of freedon of the inverse Wishart distribution.
            cov_scale (int): Scale matrix of the inverse Wishart distribution.
        """
        super().__init__(
            cluster_rate * util.volume(bounds), alpha, beta)
        
        self.cluster_rate = cluster_rate
        self.bounds = bounds
        self.cov_df = cov_df
        self.cov_scale = np.array(cov_scale)


    def log_evidence(self, x) -> float:
        """
        TODO
        """
        evidence = util.monte_carlo_integration(
            self.phi, self.bounds, int(1e6), x = x
        ) 
        
        return np.log(evidence)

    
    def phi(self, m, x):
        """
        TODO
        """
        diff_mx = m.reshape((m.shape[0], m.shape[1], 1)) - x.T
        diff_mx_transp = m.reshape((m.shape[0], 1, m.shape[1])) - x

        temp = np.matmul(
            diff_mx, diff_mx_transp, axes=[(-2, -1), (-2, -1), (1, 2)]
        )
        temp = np.linalg.det(
            self.cov_scale + temp
        )

        num = (
            np.linalg.det(self.cov_scale) ** (self.cov_df / 2.) 
            * gamma((self.cov_df + 1) / 2.)
        )

        den = (
            np.power(temp, (self.cov_df + 1) / 2.) 
            * np.pi * self.cluster_rate * util.volume(self.bounds) 
            * gamma(self.cov_df / 2.)
        )

        return num / den

"""
For tests
g = GaussianModel(4., [[0, 0], [1, 1]], 1., 1., 5, 1e-3*np.identity(2))
m = np.zeros((10, 2)) 
x = np.zeros((1, 2))
x[0, 0] = 1.2
print(np.exp(g.new_cluster_log_prob(x)))"""
