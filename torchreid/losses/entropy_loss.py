import torch
from torch import nn
from torch.nn import functional as F

class ConfidencePenalty(nn.Module):
    def __init__(self):
        super(ConfidencePenalty, self).__init__()

    def forward(self, targets,pids):
        b = F.softmax(targets, dim=1) * F.log_softmax(targets, dim=1)
        b = -1.0 * b.sum()
        return b
