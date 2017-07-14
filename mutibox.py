import torch
import torch.nn as nn

class Multibox(nn.Module):
    num_classes = 21
    num_anchors = [4,6,6,6,4,4]

    in_planes = [512,1024,512,256,256,256]
    def __init__(self):
        super(Multibox,self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.layers.append(nn.Conv2d(self.in_planes[i],self.num_anchors[i]*(4+21),kernel_size=3,padding=1))
    def forward(self,hs):
        outs = []
        for i,x in enumerate(hs):
            out = self.layers[i](x)
            out = out.permute(0,2,3,1).contiguous()
            out = out.view(out.size(0),-1,25).contiguous()
            outs.append(out)
        pred = torch.cat((outs),1)
        cls = pred[:,:,:21]
        bbox = pred[:,:,21:25]
        return cls,bbox







