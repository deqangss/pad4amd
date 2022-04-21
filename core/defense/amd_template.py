"""
abstract class of detecting adversarial malware
"""


class DetectorTemplate(object):
    def __init__(self):
        self.tau = None
        self.is_detector_enabled = True

    def forward(self, x):
        """class prediction & density estimation"""
        raise NotImplementedError

    def get_threshold(self):
        """calculate the threshold for rejecting outliers"""
        raise NotImplementedError

    def get_tau_sample_wise(self):
        raise NotImplementedError

    def indicator(self):
        """return a vector of boolean flags indicating whether reject a sample"""
        raise NotImplementedError
