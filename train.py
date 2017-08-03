import torch
from torch.autograd import Variable
from Database import IMDB
from network import SSD300
from DataProcess import dataProcess
import torch.optim as optim
import time
imdb = IMDB()
epochs = 10000
batch_size = 32
batch_per_epoch = 9963//batch_size
lr = 1e-1
vgg_model = 'vgg16-397923af.pth'
model = SSD300()
model.load_state_dict(torch.load('ssd.pth'))
# model = torch.nn.DataParallel(model,device_ids=[0])
model=model.cuda()
print "load model successfully"
model.train()

DataProcessor = dataProcess()

optimizer = optim.SGD(model.parameters(),lr = lr,weight_decay=0.00005,momentum=0.9)
for epoch in range(epochs):
    for batch in range(batch_per_epoch):
        images, Annos = imdb.getItems(batch_size)
        images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2).contiguous()
        images = images.cuda()
        images = Variable(images)



        total_loss = 0
        loc_loss = Variable(torch.Tensor([0])).cuda()
        cls_loss = Variable(torch.Tensor([0])).cuda()
        time_start = time.time()
        for i in range(batch_size):


            predcls, predbox = model(images[i].unsqueeze(0))

            Annos_ = torch.Tensor(Annos[i])
            gt_cls = Annos_[:, :1]
            gt_boxes = Annos_[:, 1:5]
            predbox_ = predbox[0]
            predcls_ = predcls[0]

            loc_loss_batch,cls_loss_batch = DataProcessor.loss(gt_boxes,gt_cls,predbox_,predcls_)
            try:
                loc_loss+=loc_loss_batch
            except:
                pass
            cls_loss+=cls_loss_batch





            # loc_loss_total+=loc_loss
            # cls_loss_total+=cls_loss

        # loc_loss_avg = loc_loss_total/batch_size
        # cls_loss_avg = cls_loss_total/batch_size
        # loss_total = loc_loss_avg+cls_loss_avg
        time_end = time.time()
        deltime = time_end - time_start
        loc_loss/=batch_size
        cls_loss/=batch_size
        loss = loc_loss + cls_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print "epoch:", epoch,"batch:",batch,"/",batch_per_epoch,  "loss:%.3f" % (loss.data[0]), "loc_loss:%.3f" % (
        loc_loss.data[0]), "cls_loss%.3f" % (cls_loss.data[0]), "time:%.2f" % (deltime)

    if epoch%10 == 0:
        torch.save(model.state_dict(),"model.pkl"+repr(epoch))

    # print "loss:",loss_total.data[0],"loc_loss:",loc_loss_avg.data[0],"cls_loss",cls_loss_avg.data[0]


