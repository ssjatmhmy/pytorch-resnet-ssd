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

class MultiBoxLayer(nn.Module):

    def __init__(self, num_classes, num_anchors, in_planes):
        super(MultiBoxLayer, self).__init__()
        self.num_anchors = num_anchors
        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        for i in range(len(in_planes)):
            self.loc_layers.append(nn.Conv2d(in_planes[i], num_anchors[i]*4, kernel_size=3, padding=1))
            self.cls_layers.append(nn.Conv2d(in_planes[i], num_anchors[i]*(num_classes+1), kernel_size=3, padding=1))
            
    def forward(self, fms):
        '''
        Args:
          fms: (list) of tensor containing intermediate layer outputs.
        Returns:
          loc_preds: (tensor) predicted locations, sized [N,H*W*#anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N,H*W*#anchors,#classes+1].
        '''
        loc_preds = []
        cls_preds = []
        for loc_layer, cls_layer, fm in zip(self.loc_layers, self.cls_layers, fms):
            loc_pred = loc_layer(fm)
            cls_pred = cls_layer(fm)
            # [N, #anchors*4,H,W] -> [N,H,W, #anchors*4] -> [N,H*W*#anchors, 4]
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 
            # [N,#anchors*(#classes+1),H,W] -> [N,H,W,#anchors*(#classes+1)] -> [N,H*W*#anchors,#classes+1]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,(self.num_classes+1))  
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

class SSD300(nn.Module):
    input_size = 300
    num_anchors = [4,6,6,6,4,4]
    in_planes = [512,1024,512,256,256,256]

    def __init__(self, depth, width=1, num_classes=20):
        super(SSD300, self).__init__()
        self.base_network = resnet_base(depth, width)
        self.norm1 = L2Norm2d(20)
        self.num_classes = num_classes
        self.in_planes = [int(round(in_plane * width)) if i < 2 else in_plane \
                          for i, in_plane in enumerate(self.in_planes)]

    def forward(self, x):
        fms = self.base_network(x)
        fms[0] = self.norm1(fms[0])
        loc_preds, conf_preds = self.multibox(fms, self.num_classes, self.num_anchors, self.in_planes)
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
