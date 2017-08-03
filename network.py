import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
from mutibox import Multibox



class L2Norm2d(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)
class SSD300(nn.Module):
    # input_size = 300


    def __init__(self):
        super (SSD300,self).__init__()
        self.base = self.VGG16()
        self.conv5_1 = nn.Conv2d(512,512,kernel_size=3,padding=1,dilation=1)
        self.conv5_2 = nn.Conv2d(512,512,kernel_size=3,padding=1,dilation=1)
        self.conv5_3 = nn.Conv2d(512,512,kernel_size=3,padding=1,dilation=1)
        self.conv_6 = nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6)

        self.conv_7 = nn.Conv2d(1024,1024,kernel_size=1)
        self.conv8_1 = nn.Conv2d(1024,256,kernel_size=1)
        self.conv8_2 = nn.Conv2d(256,512,kernel_size=3,padding=1,stride=2)

        self.conv9_1 = nn.Conv2d(512,128,kernel_size=1)
        self.conv9_2 = nn.Conv2d(128,256,kernel_size=3,padding=1,stride=2)

        self.conv10_1 = nn.Conv2d(256,128,kernel_size=1)
        self.conv10_2 = nn.Conv2d(128,256,kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.multibox = Multibox()
        self.weight_init()
    def forward(self,x):
        hs = []
        mutibox = Multibox()
        mutibox.cuda()
        h = self.base(x)

        self.norm4 = L2Norm2d(20)
        hs.append(self.norm4(h)) #block4 512*38*38

        h = F.max_pool2d(h,kernel_size=2,stride=2,ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h,kernel_size=3,padding=1,stride=1,ceil_mode=True)

        h = F.relu(self.conv_6(h))
        h = F.relu(self.conv_7(h))
        hs.append(h) # block7 1024*19*19

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)# block8 512*10*10

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)  # conv9_2 256*5*5

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)  # conv10_2 256*3*3

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))

        hs.append(h)  # conv11_2
        cls,bbox = mutibox(hs)
        return cls,bbox

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #

    def VGG16(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(True)]
                in_channels = x

        return nn.Sequential(*layers)