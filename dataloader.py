import torch.utils.data
import os
import pydicom
from torchvision import transforms
import torch
import numpy as np
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import scipy.io as sio

class ImageQualityPair(torch.utils.data.Dataset):
    def __init__(self,root,mode,resized_size=64,original_size=320,cropped_size=160):
        self.root=root
        self.mode=mode
        self.resized_size=resized_size
        self.original_size=original_size
        self.cropped_size=cropped_size
        self.im_list=[]
        i=0
        while os.path.exists(root+mode+str(i)+'/'):
            if mode=='score/':
                array=np.load(root+mode+str(i)+'/score.npy')
            for j in range(1,21):
                temp=[root+mode+str(i)+'/T2/'+str(j)+'.dcm',root+mode+str(i)+'/ADCresampled/'+str(j)+'.dcm']
                if mode=='score/':
                    temp.append(array[2])
                    temp.append(i)
                    temp.append(j)
                self.im_list.append(temp)
            i+=1

    def __len__(self):
        return len(self.im_list)

    def augment(self,t2,adc):
        minn=min(self.cropped_size*3//4,self.original_size//3)
        maxx=max(self.cropped_size*4//3,self.original_size*7//10)
        rand_size=np.random.randint(minn,maxx)
        if np.random.uniform(0,5)<3:
            rand_num=1
        else:
            rand_num=2
        trans=transforms.Compose([
            transforms.RandomRotation(15),
            transforms.CenterCrop(rand_size),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomAutocontrast(0.2),
            transforms.RandomAdjustSharpness(rand_num,0),
        ]
        )
        gamma=np.random.uniform(0.5,1.4)
        seed=np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        t2=trans(t2)
        random.seed(seed)
        torch.manual_seed(seed)
        adc=trans(adc)
        t2=transforms.ToTensor()(np.array(t2))
        t2=t2**gamma
        adc=transforms.ToTensor()(np.array(adc))
        adc=adc**gamma
        return t2,adc

    def __getitem__(self,item):
        t2=pydicom.dcmread(self.im_list[item][0])
        adc=pydicom.dcmread(self.im_list[item][1])
        t2=t2.pixel_array
        adc=adc.pixel_array
        t2=Image.fromarray((t2-t2.min())/((t2.max()-t2.min())))
        adc=Image.fromarray(np.clip(adc,0,1600)/1600)
        if self.mode=='no_score/':
            t2,adc=self.augment(t2,adc)
            score=0
            folder='-1/'
            im_num='0'
        else:
            score=self.im_list[item][2]
            folder=str(self.im_list[item][3])+'/'
            im_num=str(self.im_list[item][4])
            t2=transforms.CenterCrop(self.cropped_size)(t2)
            t2=transforms.ToTensor()(t2)
            adc=transforms.CenterCrop(self.cropped_size)(adc)
            adc=transforms.ToTensor()(adc)
        t2=transforms.Resize((self.resized_size,self.resized_size))(t2)
        t2=t2*2-1
        adc=transforms.Resize((self.resized_size,self.resized_size))(adc)
        adc=adc*2-1
        return {'t2':t2,'adc':adc,'score':score,'folder':folder,'im_num':im_num}

'''
a=ImageQualityPair('../data/ProstateImageQuality1/','score/')
for i in range(20):
    dic=a.__getitem__(i)
    plt.subplot(1,2,1)
    plt.imshow(dic['t2'][0,:,:])
    plt.subplot(1,2,2)
    plt.imshow(dic['adc'][0,:,:])
    plt.show()
'''

