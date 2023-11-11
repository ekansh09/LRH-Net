import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.optim as optim
import torch
from torch.optim import lr_scheduler
import logging
import warnings
import time

from torchsummary import summary
from functools import partial
import torch.nn.functional as F


##### SEResnet - Teacher Model  #######

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,is_last = False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out

class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=12, out_channel=24, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(5, 10)
        self.fc = nn.Linear(512 * block.expansion + 10, out_channel)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """
                          
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        
        layers.append(block(self.inplanes, planes, stride, downsample,is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            
            layers.append(block(self.inplanes, planes,is_last=(i == blocks-1)))

        return nn.Sequential(*layers)



    def forward(self, x, ag, is_feat=False, preact=False):
        
        #print("input:",x.shape, ag.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x
        x = self.maxpool(x)
        f1 = x

        x, f2_pre = self.layer1(x)
        f2 = x
        x, f3_pre = self.layer2(x)
        f3 = x
        x, f4_pre = self.layer3(x)
        f4 = x
        x, f5_pre = self.layer4(x)
        f5 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f6 = x
        
        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)
        #print("x:",x.shape)
        x = self.fc(x)
        #x = self.sig(x)

        if is_feat:
            if preact:
                return [f0, f1, f2_pre, f3_pre, f4_pre, f5_pre, f6], x
            else:
                return [f0, f1, f2, f3, f4,f5,f6], x
        else:
            return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model




##### LRH-Net - Student Model  #######

def conv1d(in_planes, out_planes, kernel_size, strides=1,padding='same', bias=True):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=strides,padding=padding, bias=bias)

class Cust_SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(Cust_SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Cust_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_channels, kernel_size=8, stride=1, padding='same', bias=False, downsample=None):
        super().__init__()
        self.conv1 = conv1d(in_planes, out_channels,kernel_size, strides = (1 if not downsample else 2), padding= ('same' if not downsample else 3))
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv1d(out_channels, out_channels,kernel_size, strides = 1)
        self.se = Cust_SELayer(out_channels)
        self.downsample = downsample
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
#         print(x.size())
#         print(residual.size())
#         print(out.size())
        out += residual
        out = self.relu(out)
        out = self.bn2(out)    
        return out

class Cust_Resnet(nn.Module):
    def __init__(self, block, layers, input_channel, num_classes):
        super(Cust_Resnet, self).__init__()
        
        self.in_channel = 16
    
        self.conv1 = nn.Conv1d(input_channel, 16, kernel_size=8, padding = 'same')
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, layers[0],out_channels = 16 )
        self.layer2 = self._make_layer(block,layers[1],out_channels = 32)
        self.layer3 = self._make_layer(block,layers[2],out_channels = 64)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(5, 10)
        self.fc1 = nn.Linear(64+10, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def _make_layer(self, block, num_residual_block, out_channels):
        downsample = None
        layers = []

        if self.in_channel != out_channels * block.expansion:
                downsample = conv1d(self.in_channel, out_channels*block.expansion,kernel_size=1, strides=2, padding=0)

        layers.append( block(in_planes=self.in_channel,out_channels =out_channels,stride=1,downsample=downsample))
        self.in_channel = out_channels * block.expansion
        for i in range(1, num_residual_block):
            layers.append(block(self.in_channel, out_channels))
        return nn.Sequential(*layers)

    def forward(self,x,ag):
        x= self.conv1(x)
        x= self.relu(x)
        x= self.layer1(x)
        x= self.layer2(x)
        x= self.layer3(x)
        x= self.avg_pool(x)
        x = x.view(x.size(0), -1)
        ag = self.fc(ag)
        x = torch.cat((ag, x), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
#         x = self.sigmoid(x)
        return x


def CustomResnet(input_channel, num_classes=24):
    return Cust_Resnet(Cust_BasicBlock, [1,1,1], input_channel, num_classes)
