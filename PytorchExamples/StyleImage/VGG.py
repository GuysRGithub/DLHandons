import torch
from collections import namedtuple
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pre_trained_features = models.vgg16(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pre_trained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pre_trained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pre_trained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pre_trained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        vgg_outputs = namedtuple("VggOutputs",
                                 ['relu1', 'relu2', 'relu3', 'relu4'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4)
        return out
