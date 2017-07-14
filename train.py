import torch
from torch.autograd import Variable
from Database import IMDB
from network import SSD300
from DataProcess import dataProcess
import torch.optim as optim
imdb = IMDB()
epochs = 10000
batch_size = 1
lr = 1e-3
vgg_model = 'vgg16-397923af.pth'
model = SSD300()

model.load_state_dict(torch.load('ssd.pth'))
print "load model successfully"
model.train()
DataProcessor = dataProcess()
optimizer = optim.SGD(model.parameters(),lr = lr,weight_decay=0.00005,momentum=0.9)

for epoch in range(epochs):
    print "epoch",epoch
    images, Annos = imdb.getItems(batch_size)
    images = torch.from_numpy(images)
    images = images.permute(0, 3, 1, 2).contiguous()
    images = Variable(images)
    predcls, predbox = model(images)

    loc_loss_total = 0
    cls_loss_total = 0
    for i in range(batch_size):
        Annos_ = torch.Tensor(Annos[i])
        gt_cls = Annos_[:, :1]
        gt_boxes = Annos_[:, 1:5]
        predbox_ = predbox[i]
        predcls_ = predcls[i]
        loc_loss,cls_loss = DataProcessor.loss(gt_boxes,gt_cls,predbox_,predcls_)
        loss = loc_loss+cls_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loc_loss_total+=loc_loss
        cls_loss_total+=cls_loss
    loc_loss_avg = loc_loss_total/batch_size
    cls_loss_avg = cls_loss_total/batch_size
    loss_total = loc_loss_avg+cls_loss_avg
    if epoch%1000 == 0:
        torch.save(model.state_dict(),"model.pkl"+repr(epoch))

    print "loss:",loss_total.data[0],"loc_loss:",loc_loss_avg.data[0],"cls_loss",cls_loss_avg.data[0]


