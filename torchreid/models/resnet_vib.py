from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['ResNet50_vib']

class ResNet50_noB(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50_noB, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50_vib(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50_vib, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.encoder = nn.Linear(2048, 2048)
        self.classifier = nn.Linear(1024, num_classes)
        self.K = 1024

    def forward(self, x, num_sample=1):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        encoding = self.encoder(f)
        mu = encoding[:,:self.K]
        std = F.softplus(encoding[:,self.K:]-5,beta=1)


        if not self.training:
            return mu, std

        encoding = self.reparametrize_n(mu,std,num_sample)
        logit = self.classifier(encoding)



        if self.loss == {'xent'}:
            return (mu, std), logit
        elif self.loss == {'xent', 'htri'}:
            return (mu, std), logit, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

    def reparametrize_n(self, mu, std, n=1):
    # reference :
    # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = std.data.new(std.size()).normal_()

        return mu + eps * std
