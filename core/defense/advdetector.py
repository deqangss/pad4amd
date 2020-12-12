import torch
import torch.nn as nn
import torch.nn.functional as F

from core.defense.maldetector import MalwareDetector, MalGAT


class AdversarialMalwareDetector(MalwareDetector):
    raise NotImplementedError