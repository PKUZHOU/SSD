import torch
import torch.nn as nn

class SSD300(nn.Module):
    input_size = 300
    def __init__(self):
        super (SSD300,self).__init__()
        self.base = self.vgg16()
        self.conv5_1 = nn.Conv2d(512,512,kernel_size=3,padding=1)

    #
    def vgg16(batch_norm=True):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                if batch_norm:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(True)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(True)]
            in_channels = v
        return nn.Sequential(layers)