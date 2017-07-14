import os
import xml.dom.minidom
import cv2
import numpy as np
import re
class IMDB:
    def __init__(self):
        self.ImgPath = '/home/zhou/PycharmProjects/SSD/VOC2007/VOC2007/JPEGImages/'
        self.AnnoPath = '/home/zhou/PycharmProjects/SSD/VOC2007/VOC2007/Annotations/'
        self.imagelist = os.listdir(self.ImgPath)
        self.Annos =[]
        self.classes = {
            'aeroplane':1,'bicycle':2,'bird':3,'boat':4,'bottle':5,'bus':6,'car':7,'cat':8,'chair':9,
        'cow':10,
        'diningtable':11,
        'dog':12,
        'horse':13,
        'motorbike':14,
        'person':15,
        'pottedplant':16,
        'sheep':17,
        'sofa':18,
        'train':19,
        'tvmonitor':20,
        'backgroud':0
        }

    def convertAnnos(self):
        f = open('Annos.txt','w')
        for image in self.imagelist:
            union = []
            image_pre,ext = os.path.splitext(image)
            xmlfile = self.AnnoPath+image_pre+'.xml'
            imgfile = self.ImgPath + image
            img = cv2.imread(imgfile)
            xratio = img.shape[1]/300.
            yratio = img.shape[0]/300.
            DomeTree = xml.dom.minidom.parse(xmlfile)
            annotations = DomeTree.documentElement
            objectlist = annotations.getElementsByTagName('object')

            for object in objectlist:
                namelist = object.getElementsByTagName('name')
                objectname = namelist[0].childNodes[0].data
                objectNum = self.classes[objectname]
                bndbox = object.getElementsByTagName('bndbox')
                for box in bndbox:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(float(x1_list[0].childNodes[0].data)/xratio)
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(float(y1_list[0].childNodes[0].data)/yratio)
                    x2_list = box.getElementsByTagName('xmax')
                    x2 = int(float(x2_list[0].childNodes[0].data)/xratio)
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(float(y2_list[0].childNodes[0].data)/yratio)
                    annos = [objectNum,x1,y1,x2,y2]
                    union.append(annos)
            f.write(image_pre+' ')
            f.write(str(union)+'\n')
            print 'covered'+repr(image_pre)
        f.close()
    def convertImages(self):
        os.chdir('IMages')
        for image in self.imagelist:
            imgfile = self.ImgPath + image
            img = cv2.imread(imgfile)
            try:
                img = cv2.resize(img,(300,300))
            except:
                pass

            cv2.imwrite(image,img)
            print 'convered'+image
    def getItems(self,num):
        ImgPath = '/home/zhou/PycharmProjects/SSD/IMages/'
        annosPath = '/home/zhou/PycharmProjects/SSD/'
        if len(self.Annos)==0:
            f = open(annosPath + 'Annos.txt', 'r')
            self.Annos=(f.readlines())
            f.close()
        lenth = len(self.Annos)
        indexs = np.random.randint(0,lenth,num)
        Images = np.zeros([num,300,300,3],dtype=np.float32)
        Boxes = []
        for i,ind in enumerate(indexs):
            Anno= self.Annos[ind].strip('\n')
            ImgNum=Anno[0:6]
            box = eval(Anno[7:])
            Image = cv2.imread(ImgPath+ImgNum+'.jpg')
            Image = Image/255.
            Images[i] = Image
            Boxes.append(box)
        return Images,Boxes




if __name__ == '__main__':
    imdb = IMDB()
    images,boxes = imdb.getItems(10)