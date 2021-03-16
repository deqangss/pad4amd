"""
abstract class of dense estimator
"""


class DenseEstimator(object):
    def __init__(self):
        self.tau = None

    def forward_g(self):
        """density estimation"""
        raise NotImplementedError

    def get_threshold(self):
        """calculate the threshold for rejecting outliers"""
        raise NotImplementedError

    def indicator(self):
        """return a vector of boolean flags indicating whether reject a sample"""
        raise NotImplementedError
