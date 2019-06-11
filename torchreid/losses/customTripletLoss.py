from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class TripletLoss_custom(nn.Module):
    """TODO: make sure each anchor use their relative positive image
    """
    def __init__(self, margin=0.3):
        super(TripletLoss_custom, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs_a, inputs_p, targets_a, targets_p):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        m, n = inputs_a.size(0), inputs_p.size(0)
        dist = torch.pow(inputs_a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(inputs_p, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, inputs_a, inputs_p.t())
        dist = dist.clamp(min=1e-12).sqrt()

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets_a.expand(m, m).eq(targets_p.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(m):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class SoftTripletLoss_custom(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """
    def __init__(self):
        super(SoftTripletLoss_custom, self).__init__()
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, inputs_a, inputs_p, targets_a, targets_p):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        m, n = inputs_a.size(0), inputs_p.size(0)
        dist = torch.pow(inputs_a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(inputs_p, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, inputs_a, inputs_p.t())
        dist = dist.clamp(min=1e-12).sqrt()

        # Compute pairwise distance, replace by the official when merged
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets_a.expand(m, m).eq(targets_p.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(m):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        loss = self.softplus(dist_ap - dist_an).sum()
        return loss
