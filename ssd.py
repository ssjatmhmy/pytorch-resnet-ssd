import torch
import torch.nn as nn

from resnet_base import resnet_base
from torch.autograd import Variable

class L2Norm2d(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)


class SSD300(nn.Module):
    input_size = 300
    num_anchors = 9
    
    def __init__(self, depth, width=1, num_classes=20):
        super(SSD300, self).__init__()
        self.base_network = resnet_base(depth, width)
        self.num_classes = num_classes

    def forward(self, x):
        fms = self.base_network(x)
        loc_preds, conf_preds = self.multibox(fms)
        return loc_preds, conf_preds
        
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    net = SSD300(depth=18, width=1)
    loc_preds, cls_preds = net(Variable(torch.randn(2,3,300,300)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

# test()
