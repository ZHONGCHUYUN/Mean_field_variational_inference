import numpy as np


def generate_data(N=1000, D=2, K=None, mu=None, sigma=1., dtype=np.float64):
    """
    Generates samples from a Mixture of Gaussian (MoG).
    Do not change this function!
    Parameters
    ----------
    N: int
       number of samples
    K: int
       number of components
    mu: vector of shape (K) or array_like of shape (K x D)
        mean of components
    sigma: float
           standard deviation of each Gaussian component

    Returns
    -------
    samples: ndarray of shape(N x D)
             Samples from MoG distribution.
    c: 1D ndarray of shape (N,)
       Cluster assignment corresponding to samples
    """
    if mu is None:
        assert K is not None, "Must pass in either mu or K"
        mu = np.random.normal(np.zeros((K, D)), np.ones((K, D)) * 2.)
    else:
        K = mu.shape[0]

    mu = mu.reshape((1, K, -1)).repeat(N, axis=0)
    c = np.random.choice(K, N)
    mu_c = mu[np.arange(N), c, :]
    samples = np.random.normal(mu_c, np.ones_like(mu_c) * sigma)
    return samples, c


class GMM_CAVI(object):
    """
    Implements the Gaussian Mixture Model (GMM) with
    Coordinate Ascent Variational Inference. The means of mixture
    components are assumed to have a Gaussian prior N(0, pmu_var),
    the cluster indicators are assumed to have a uniform prior and
    the likelihood variance term is assumed to be known and fixed
    as the identity matrix.
    The variational distribution is specified by a gaussian
    for the means of the components and a categorical distribution
    over the mixture components:
    ----------
    m: mean of gaussian of the mixture components q(mu) - K x D
    s2: variance of gaussian of the mixture components q(mu) - K x D
    pi: probabilities for the categorical over mixture components q(c) - N x K
    ----------
    Initialisation Parameters
    ----------
    X: float,
       D dimensional dataset
    K: int,
       number of mixture components
    pmu_var: float,
       prior variance on mean of components p(mu)
    ----------
    """

    def __init__(self, X, K, pmu_var=1., seed=31415):
        """
        Initialization of (variational) parameters.
        Do not change this function!
        """
        self.X = X
        self.K = K
        self.N, self.D = X.shape

        self.rng = np.random.RandomState(seed)

        self.pmu_var = np.float64(self.D * [pmu_var])

        self.m = np.zeros(
            (self.K, self.D)) + self.rng.randn(self.K, self.D) * 1e-2
        self.s2 = np.ones_like(self.m)

        self.pi = self.rng.dirichlet(
            self.rng.randint(1, 10, size=self.K), self.N)

    def update_m(self):
        """
        Returns the optimal factor for the mean of
        the variational distibution over the cluster means q*(mu_k)
        - Ignoring constant terms!
        -------
        m: K x D
        """
        m = None
        #######################################################################
        # TODO: Implement the update for the mean of q*(mu_k) here using      #
        #       the appropriate class variables.                              #
        #######################################################################
        K = self.K
        D = self.D
        N = self.N
        m = np.zeros((K,D))
        pi = self.pi
        X = self.X
        pmu_var = self.pmu_var

        for k in range(K):
            numerator = 0
            denominator = 0
            for i in range(N):
                numerator = numerator + pi[i][k]*X[i]
                denominator = denominator + pi[i][k]
            m[k] = numerator/(denominator+1/pmu_var[0])

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return m

    def update_s2(self):
        """
        Returns the optimal factor for the variance of
        the variational distibution over the cluster means q*(mu_k)
        - Ignoring constant terms!
        -------
        s2: K x D
        """
        s2 = None
        #######################################################################
        # TODO: Implement the update for the variance of q*(mu_k) here using  #
        #       the appropriate class variables.                              #
        #######################################################################
        K = self.K
        D = self.D
        pi = self.pi
        s2 = np.zeros((K,D))
        pmu_var = self.pmu_var
        one_vec = np.ones((1,D))

        for k in range(K):
                s2[k] = 1/(sum(pi[:,k])*one_vec + 1/pmu_var)
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return s2

    def update_pi(self):
        """
        Returns the *normalised* optimal factor for the probabilities
        of the mixture components q(c_i) - Ignoring constant terms!
        -------
        pi: N x K
        """
        pi = None
        #######################################################################
        # TODO: Implement the update for the probabilities                    #
        #       of q(c_i). Return the normalised distribution.                #
        #######################################################################
        K = self.K
        N = self.N
        pi = np.zeros((N,K))
        m = self.m
        s2 = self.s2
        X = self.X

        for k in range(K):
            for i in range(N):
                pi[i][k] = np.exp(-0.5*sum(s2[k]) - 0.5*m[k].T.dot(m[k]) + m[k].T.dot(X[i]) - 1)

        #normalization
        for i in range(N):
            pi[i] = pi[i] / np.sum(pi[i])

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        return pi

    def log_joint(self):

        """
        Returns the expected log probability density of the joint model
        log p(X, mu, c) under q(mu, c) - Ignoring constant terms!
        -------
        log_density: scalar
        """
        log_density = None
        #######################################################################
        # TODO: Implement the expectation of the log density of the model     #
        #       here and return the sum - ignoring any constant terms         #
        #######################################################################

        m = self.m
        X = self.X
        pi = self.pi
        s2 = self.s2
        pmu_var = self.pmu_var
        N = self.N
        K = self.K

        log_density = 0
        #expectation
        for i in range(N):
            for k in range(K):
                log_density = log_density + pi[i][k]* (m[k].T.dot(X[i]) - 0.5 * (m[k].T.dot(m[k]) + np.sum(s2[k])))
        for k in range(K):
            log_density = log_density - 0.5 * m[k].T.dot(np.linalg.inv(np.diagflat(pmu_var))).dot(m[k]) - 0.5 * np.sqrt(s2[k]).T.dot(np.linalg.inv(np.diagflat(pmu_var))).dot(np.sqrt(s2[k]))

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return log_density

    def log_var(self):

        """
        Returns the expected log probability density of the variational distribution
        log q(mu, c) under q(mu, c) - Ignoring constant terms!
        -------
        log_q_density: scalar
        """
        log_q_density = None
        #######################################################################
        # TODO: Implement the expectation of the log variational density here #
        #       and return the sum - ignoring any constant terms              #
        #######################################################################
        N = self.N
        K = self.K
        D = self.D
        s2 = self.s2
        m = self.m
        pi = self.pi
        log_q_density = 0
        for k in range(K):
            for i in range(N):
                log_q_density = log_q_density + pi[i][k]*np.log(pi[i][k])
            log_q_density = log_q_density - 0.5 * np.log(np.linalg.det(np.diagflat(s2[k])))
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return log_q_density

    def elbo(self):
        """
        Returns the evidence lower bound (ELBO) using log_joint and log_var
        -------
        elbo: scalar
        """
        elbo = None
        #######################################################################
        # TODO: Implement the elbo using log_joint and log_var from before    #
        #######################################################################
        elbo = self.log_joint() - self.log_var()
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        return elbo

    def cavi(self):
        """
        Updates variational parameters
        Do not change this function.
        """
        self.pi = self.update_pi()
        assert self.pi is not None, "Update for pi not implemented"
        self.s2 = self.update_s2()
        assert self.s2 is not None, "Update for s2 not implemented"
        self.m = self.update_m()
        assert self.m is not None, "Update for m not implemented"

    def fit(self, max_iter=1000, threshold=1e-10):
        """
        Performs CAVI using the optimal factor updates.
        Do not change this function.
        """

        elbo_trace = []
        for iter in range(max_iter):
            self.cavi()
            elbo_trace.append(self.elbo())

            if iter > 0:
                delta_elbo = elbo_trace[-1] - elbo_trace[-2]
                if delta_elbo < threshold:
                    break

        return np.float64(elbo_trace)
