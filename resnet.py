import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from mmcv.cnn import DeformConv2d  # 导入可变形卷积

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, deformable=False):
    if deformable:
        return DeformConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last_block=False, deformable=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride, deformable)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.is_last_block = is_last_block
        self.deformable = deformable
        
        if self.is_last_block:
            self.se = SEBlock(planes)
        else:
            self.se = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se is not None:
            out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, strides, compress_layer=True):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1_new = nn.Conv2d(3, 32, kernel_size=3, stride=strides[0], padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Track which block is the last in layer1 and use deformable in layer4
        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[1], is_last_block=True, deformable=False)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[2], deformable=False)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[3], deformable=False)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[4], deformable=True)  # 使用可变形卷积
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[5], deformable=False)

        self.compress_layer = compress_layer        
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_last_block=False, deformable=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        # Determine which block is the last one in the layer
        last_block_idx = blocks - 1
        for i in range(blocks):
            is_last = (i == last_block_idx) and is_last_block
            layers.append(block(self.inplanes, planes, stride if i == 0 else 1, 
                               downsample if i == 0 else None, is_last, deformable))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, multiscale=False):
        out_features = []
        x = self.conv1_new(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_shape = x.size()[2:]
        x = self.layer1(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer3(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer4(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer5(x)
        if not self.compress_layer:
            out_features.append(x)
        else:
            if x.size()[2:] != tmp_shape:
                tmp_shape = x.size()[2:]
                out_features.append(x)
            x = self.layer6(x)
            out_features.append(x)
        return out_features

def resnet45(strides, compress_layer):
    model = ResNet(BasicBlock, [3, 4, 6, 6, 3], strides, compress_layer)
    return model