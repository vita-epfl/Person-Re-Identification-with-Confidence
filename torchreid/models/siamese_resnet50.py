from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

import pdb
__all__ = ['Siamese_Resnet50']

class EltwiseSubEmbed(nn.Module):
    def __init__(self, nonlinearity='square', use_batch_norm=False,
                 use_classifier=False, num_features=0, num_classes=0):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.use_classifier:
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifier:
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
        else:
            x = x.sum(1)
        return x

class Siamese_Resnet50(nn.Module):
    def __init__(self, num_classes,num_instances, loss={'xent'}, **kwargs):
        super(Siamese_Resnet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.instances_num = num_instances
        self.classifier_sub = EltwiseSubEmbed(nonlinearity='square', use_batch_norm=True,
                     use_classifier=True, num_features=2048, num_classes=num_classes)

    def forward(self, x):
        x = self.base(x)

        if not self.training:
            return F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        N, C, H, W = x.size()
        probe_num = int(N / self.instances_num)
        gallery_num = int(N - N / (self.instances_num))
        x = x.view(probe_num, self.instances_num, C, H, W)

        probe_x = x[:, 0, :, :, :]
        probe_x = probe_x.contiguous()
        probe_x = probe_x.view(probe_num, C, H, W)
        gallery_x = x[:, 1:self.instances_num, :, :, :]
        gallery_x = gallery_x.contiguous()
        gallery_x = gallery_x.view(gallery_num, C, H, W)
        N_probe, C, H, W = probe_x.size()

        N_gallery = gallery_x.size(0)
        probe_x = probe_x.unsqueeze(1)
        probe_x = probe_x.expand(N_probe, N_gallery, C, H, W)
        probe_x = probe_x.contiguous()
        gallery_x = gallery_x.unsqueeze(0)
        gallery_x = gallery_x.expand(N_probe, N_gallery, C, H, W)
        gallery_x = gallery_x.contiguous()
        probe_x = probe_x.view(N_probe * N_gallery, C, H, W)

        gallery_x = gallery_x.view(N_probe * N_gallery, C, H, W)
        probe_x = F.avg_pool2d(probe_x, probe_x.size()[2:])
        gallery_x = F.avg_pool2d(gallery_x, gallery_x.size()[2:])
        probe_x = probe_x.view(probe_x.size(0), -1)
        gallery_x = gallery_x.view(gallery_x.size(0), -1)

        y = self.classifier_sub(probe_x,gallery_x)

        if self.loss == {'xent'}:
            return y
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
