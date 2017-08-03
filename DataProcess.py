import torch
import numpy as np
from torch.autograd import Variable
import time
import random
from bbious import build_tg,bbox_ious
import torch.nn.functional as F
class dataProcess:
    def __init__(self):
        scale = 300.
        steps = [s / scale for s in (8, 16, 32, 64, 100, 300)]
        sizes = [s / scale for s in (30., 60., 111., 162., 213., 264., 315.)]
        aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
        feature_map_sizes = (38, 19, 10, 5, 3, 1)
        offset = 0.5
        numlayers = len(feature_map_sizes)
        DefBoxes = []
        for layer in range(numlayers):
            fmsize = feature_map_sizes[layer]
            size = sizes[layer]
            for w in range(fmsize):
                for h in range(fmsize):
                    x = steps[layer]*(h+offset)
                    y = steps[layer]*(w+offset)
                    s = size
                    DefBoxes.append([x,y,s,s])
                    s = np.sqrt(sizes[layer]*sizes[layer+1])
                    DefBoxes.append([x,y,s,s])
                    for ar in aspect_ratios[layer]:
                        s1 = size*np.sqrt(ar)
                        s2 = size/np.sqrt(ar)
                        DefBoxes.append([x,y,s1,s2])
                        DefBoxes.append([x,y,s2,s1])
        self.DefBoxes = torch.Tensor(DefBoxes)
        Defboxes_c = torch.Tensor(self.DefBoxes.size())
        Defboxes_c[:, 0] = self.DefBoxes[:, 0] - self.DefBoxes[:, 2] * 0.5
        Defboxes_c[:, 1] = self.DefBoxes[:, 1] - self.DefBoxes[:, 3] * 0.5
        Defboxes_c[:, 2] = self.DefBoxes[:, 0] + self.DefBoxes[:, 2] * 0.5
        Defboxes_c[:, 3] = self.DefBoxes[:, 1] + self.DefBoxes[:, 3] * 0.5
        self.Defboxes_c = Defboxes_c.numpy().astype(np.float)
        #DefBoxes : (x,y,w,h)
    def ious(self,box1,box2):

        #defbox : [8732,4] xmin,ymin,xmax,ymax
        #box : [,4] xmin,ymin,xmax,ymax
        '''num_box1 = box1.size(0)
        num_box2 = box2.size(0)
        bbious = torch.Tensor(num_box1,num_box2)
        for x in range(num_box1):
            for y in range(num_box2):
                w1 = box1[x][2]-box1[x][0]
                h1 = box1[x][3]-box1[x][1]
                s1 = w1 * h1
                w2 = box2[y][2] - box2[y][0]
                h2 = box2[y][3] - box2[y][1]
                s2 = w2 * h2
                Uw = min(box1[x][2],box2[y][2])-max(box1[x][0],box2[y][0])
                Uw = max(Uw,0)
                Uh = min(box1[x][3],box2[y][3])-max(box1[x][1],box2[y][1])
                Uh = max(Uh,0)
                U = Uw*Uh
                try:
                    iou = (U/(s1+s2-U))
                except:
                    iou = 0
                bbious[x][y]=iou'''

        bbious = bbox_ious(
        np.ascontiguousarray(box1, dtype=np.float),
        np.ascontiguousarray(box2, dtype=np.float)
        )

        return bbious


    def loss(self,gtboxes,gtclses,predbox,pred_cls):
        #gtboxes :(,4)xmin,ymin,xmax,ymax
        none_pos = False
        scale = 300.
        gtboxes =  gtboxes/scale
        gtboxes = gtboxes.numpy().astype(np.float)
        gtclses = gtclses.numpy().astype(np.float)

        #convert cx cy w h to xmin ymin xmax ymax


        ious = self.ious(gtboxes,self.Defboxes_c)
        #ious : N*8732
        ious_ = torch.from_numpy(ious)
        pos_mask = ious_>0.5
        pos_num = pos_mask.view(-1).long().sum()
        pos_mask = pos_mask.numpy()
        if pos_num==0:
            pos_num = 1
            none_pos = True
        neg_num = int(3*pos_num)
        #calculate the numbers of pos and neg



        DefBoxes_np = self.DefBoxes.numpy().astype(np.float)
        '''for i in range(ious.shape[0]):
            xmin, ymin, xmax, ymax = gtboxes[i][0], gtboxes[i][1], gtboxes[i][2], gtboxes[i][3]
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + 0.5 * w
            y = ymin + 0.5 * h
            cls = gtclses[i]
            indice = np.nonzero(pos_mask[i])
            if(len(indice)!=0):
                for ind in indice:
                    defboxes = self.DefBoxes[ind,:]
                    tgboxes[ind][0]= (x-defboxes[0])/defboxes[2]
                    tgboxes[ind][1]= (y-defboxes[1])/defboxes[3]
                    tgboxes[ind][2]= np.log(w/defboxes[2])
                    tgboxes[ind][3]= np.log(h/defboxes[3])
                    tgcls[ind][int(cls[0])]=1
            else:
                pass'''
        tg = build_tg(gtclses,gtboxes,ious,pos_mask,DefBoxes_np).astype(np.float32)
        tgboxes = tg[:,:4]
        tgcls = tg[:,4:]

        #build the pos target
        tgboxes = torch.from_numpy(tgboxes)
        tgcls = torch.from_numpy(tgcls)
        pos_mask = (torch.sum(tgcls,dim=1)>0)
        pos_inds = torch.nonzero(pos_mask)

        try:
            pos_inds = [int(ind[0]) for ind in pos_inds]
        except:
            pos_inds = []



        for x in np.random.randint(0,pos_mask.size(0),neg_num):
            if len(pos_inds)==0:
                tgcls[x][0]=1
            else:
                if x in pos_inds:
                    pass
                else:
                    tgcls[x][0]=1


        box_mask = pos_mask.expand_as(predbox)
        box_mask = box_mask.cuda()
        cls_mask = (torch.sum(tgcls,dim=1)>0).expand_as(pred_cls)
        cls_mask = cls_mask.cuda()


        # predbox = predbox.cpu()
        # pred_cls = pred_cls.cpu()
        tgboxes = tgboxes.cuda()
        tgboxes = tgboxes[box_mask].view(-1, 4)
        tgcls = tgcls.cuda()
        tgcls = tgcls[cls_mask].view(-1,21)
        predbox = predbox[box_mask].view(-1,4)
        pred_cls = pred_cls[cls_mask].view(-1,21)
        tgboxes = Variable(tgboxes)
        tgcls = Variable(tgcls)
        _,tgcls = tgcls.max(1)
        tgcls = tgcls.squeeze(1)

        loc_loss = F.smooth_l1_loss(predbox,tgboxes)
        if none_pos:
            loc_loss = Variable(torch.Tensor([0])).cuda()
        cls_loss = F.cross_entropy(pred_cls,tgcls)
        import math
        if math.isnan(loc_loss.data[0]):
            print loc_loss[0]
        else:
            pass
        return loc_loss,cls_loss












    def dataDecoder(self,boxes):
        #return the predicted loccation imformation in foramt of xmin,ymin,xmax,ymax

       #boxes : [N,8732,4], [tx,ty,tw,th]
        scale = 300
        '''N = boxes.size(0)
        defboxes = self.DefBoxes.expand_as(boxes)
        tx,ty,tw,th = boxes[:,:,0],boxes[:,:,1],boxes[:,:,2],boxes[:,:,3]
        cx = tx.data*defboxes[:,:,2]
        cy = ty.data*defboxes[:,:,3]
        x = cx+ defboxes[:,:,0]
        y = cy+ defboxes[:,:,1]
        w = torch.exp(tw.data)*defboxes[:,:,2]
        h = torch.exp(ty.data)*defboxes[:,:,3]
        xmin = x-0.5*w
        ymin = y-0.5*h
        xmax = x+0.5*w
        ymax = y+0.5*h
        xmin = torch.unsqueeze(xmin,2)
        ymin = torch.unsqueeze(ymin,2)
        xmax = torch.unsqueeze(xmax,2)
        ymax = torch.unsqueeze(ymax,2)
       
       #return xmin,ymin,xmax,ymax divided by scale 300 

        return torch.cat(([xmin,ymin,xmax,ymax]),2)'''
        boxes = boxes.data
        tx, ty, tw, th = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        cx = tx * self.DefBoxes[:,2]
        cy = ty * self.DefBoxes[:,3]
        x = cx + self.DefBoxes[:,0]
        y = cy + self.DefBoxes[:,1]
        w = torch.exp(tw) * self.DefBoxes[:,2]
        h = torch.exp(ty) * self.DefBoxes[:,3]
        xmin = x - 0.5 * w
        ymin = y - 0.5 * h
        xmax = x + 0.5 * w
        ymax = y + 0.5 * h

        xmin = torch.unsqueeze(xmin, 1)
        ymin = torch.unsqueeze(ymin, 1)
        xmax = torch.unsqueeze(xmax, 1)
        ymax = torch.unsqueeze(ymax, 1)
        return torch.cat(([xmin,ymin,xmax,ymax]),1)

if __name__ == '__main__':
    dp = dataProcess()