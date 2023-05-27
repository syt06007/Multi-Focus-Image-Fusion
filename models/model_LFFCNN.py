import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np
import copy

feature_map_fusion = []
save_outputs=[]
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.padding = (1,1,1,1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class LFFCNN(nn.Module):
    def __init__(self):
        super(LFFCNN, self).__init__()
    
        resnet = models.resnet101(pretrained=True)
        resnet_low = models.resnet101(pretrained=True)

        for p in resnet.parameters():
            p.requires_grad = False
        
        for pp in resnet_low.parameters():
            pp.requires_grad = False


        self.conv1 = resnet.conv1 
        self.conv2 = ConvBlock(64, 64)
        self.conv1_1 = resnet_low.conv1
        self.conv1_2 = ConvBlock(64, 64)
        self.bilinear = nn.UpsamplingBilinear2d(scale_factor=3)
        self.conv3 = ConvBlock(128, 64)

        self.conv4 = ConvBlock(64, 32)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv6 = nn.Conv2d(3, 3, kernel_size=1, padding=0, stride=1, bias=True)
        
        self.feature_map_fusion = feature_map_fusion

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))        
        
        for p in resnet.parameters():
            p.requires_grad = False

        self.conv1.stride = 1
        self.conv1.padding = (0, 0)
        
        self.conv1_1.stride = 3
        self.conv1_1.padding = 0



    def tensor_max(self, tensors): #fuse_scheme : elementwise_maximum
        max_tensor = None
        for i, tensor in enumerate(tensors): # enumerate(tensors[0])를 통해 
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor) # Returns the maximum value of all elements in the input tensor            
        return max_tensor

    def tensor_cat(self, tensors, low_tensors):
        out_tensors = []
        for i in range(len(tensors)):
            cat_tensor = torch.cat((tensors[i], low_tensors[i]), dim = 1) # channel direction concatenate
            out_tensors.append(cat_tensor)
        return out_tensors

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors
        
    def tensor_padding(self, tensors, padding=(1,1,1,1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors[0]:
            out_tensor = F.pad(tensor, padding, mode = mode, value = value)
            out_tensors.append(out_tensor)
        return out_tensors
# tensors[0]
    
    def forward(self, *tensors): 
        tensors_low = copy.deepcopy(tensors)
        # Feature extraction
        outs = self.tensor_padding(tensors = tensors, padding=(3,3,3,3), mode = 'replicate')
        outs_low = self.tensor_padding(tensors = tensors_low, padding=(2,2,2,2), mode = 'replicate')

        # Feature extraction
        outs = self.operate(self.conv1, outs) # resnet101.conv1
        outs = self.operate(self.conv2, outs)
        outs_low = self.operate(self.conv1_1, outs_low)
        # outs_low = self.operate(self.conv1_2, outs_low)
        outs_low = self.operate(self.bilinear, outs_low)
        outs = self.tensor_cat(outs, outs_low)
        outs = self.operate(self.conv3, outs)

        # Feature Fusion (Fusion Layer)
        out = self.tensor_max(outs)
        # feature_map_fusion.append(out)

        # Feature reconstruction
        out = self.conv4(out)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.conv5(out)
        # out = F.dropout(out, p=0.1, training=self.training)
        # out = self.conv6(out)


        return out

    def feature_map_fn(self):
        return feature_map_fusion
    def call_save_outputs(self):
        return save_outputs