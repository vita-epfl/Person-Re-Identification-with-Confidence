from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class AngularLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self,num_classes, epsilon=0.1, use_gpu=True, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, targets):
        self.it += 1
        cos_theta,phi_theta = input
        target = targets.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = self.logsoftmax(output)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        #logpt = logpt.gather(1,targets)
        # logpt = (targets * logpt).sum(0)
        # logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())

        loss = -1 * (targets * logpt)
        loss = loss.mean(0).sum()

        return loss

class AdaptiveLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    """
    def __init__(self, num_classes, use_gpu=True):
        super(AdaptiveLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, epsilon):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        epsilon = epsilon.data.squeeze()
        if self.use_gpu: targets = targets.cuda()
        for i in range(targets.size(0)):
            targets[i] = (1 - epsilon[i]) * targets[i] + epsilon[i] / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class AdaptiveLabelSmooth_sigmoid(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    """
    def __init__(self, num_classes, use_gpu=True):
        super(AdaptiveLabelSmooth_sigmoid, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, epsilon):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        targets = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        epsilon = epsilon.data.squeeze()
        if self.use_gpu: targets = targets.cuda()
        for i in range(targets.size(0)):
            targets[i] = (1 - epsilon[i]) * targets[i] + epsilon[i] / self.num_classes
        loss = self.loss(inputs, targets)
        return loss


class modifiedBCE(nn.Module):
    def __init__(self, use_gpu=True):
        super(modifiedBCE, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.use_gpu = use_gpu
    def forward(self, inputs, targets):
        targets = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        loss = self.loss(inputs, targets)
        return loss

class LabelSmooth_sigmoid(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes.
    """
    def __init__(self, num_classes,epsilon=0.1, use_gpu=True):
        super(LabelSmooth_sigmoid, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.epsilon = epsilon
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        targets = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = self.loss(inputs, targets)
        return loss
