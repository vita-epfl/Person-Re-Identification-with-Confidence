import torch
from torch import nn
from torch.nn import functional as F
import pdb
import sys

class JSD_loss(nn.Module):
    def __init__(self,num_classes):
        super(JSD_loss, self).__init__()
        self.num_classes = num_classes
        self.tensor = torch.ones((2,), dtype=torch.float)

    def forward(self, targets,pids):
        uniform = self.tensor.new_full(targets.size(), 1.0/self.num_classes,requires_grad=True).cuda()
        targets = F.softmax(targets, dim=1)
        M = 0.5*(uniform + targets)
        kl_p_m= -(targets * (M / (targets)).log()).sum()
        kl_u_m= -(uniform * (M / uniform).log()).sum()
        return 0.5*(kl_p_m + kl_u_m)
