import models

from args import TestArgParser
from dataloader import ImageQualityPair
from evaluation import evaluate

from logger import TrainLogger
from saver import ModelSaver
import torch.utils.data
import torch.nn.functional as F
import time
import numpy as np
import warnings
import os
import cv2
warnings.filterwarnings("ignore")


def get_images(args):
    # Get model
    model = models.__dict__[args.model](args)
    if args.ckpt_path:
        model = ModelSaver.load_model(model, args.model_path, args.gpu_ids, is_training=True)
    model = model.to(args.device)
    model.train()
    if not os.path.exists(args.res_path):
        os.mkdir(args.res_path)
    # Get loader, logger, and saver
    val_loader=get_data_loaders(args)

    # Train
    for _,batch in enumerate(val_loader):

        # Train over one batch
        model.set_inputs(batch['t2'],batch['adc'])
        model.test()
        tgt2src=np.moveaxis(model.tgt2src.detach().cpu().numpy(),1,-1)
        src2tgt=np.moveaxis(model.src2tgt.detach().cpu().numpy(),1,-1)
        src=np.moveaxis(model.src.detach().cpu().numpy(),1,-1)
        tgt=np.moveaxis(model.tgt.detach().cpu().numpy(),1,-1)
        for i in range(src.shape[0]):
            if not os.path.exists(args.res_path+batch['folder'][i]):
                os.mkdir(args.res_path+batch['folder'][i])
            if not os.path.exists(args.res_path+batch['folder'][i]+'T2/'):
                os.mkdir(args.res_path+batch['folder'][i]+'T2/')
            if not os.path.exists(args.res_path+batch['folder'][i]+'T2_/'):
                os.mkdir(args.res_path+batch['folder'][i]+'T2_/')
            if not os.path.exists(args.res_path+batch['folder'][i]+'ADC/'):
                os.mkdir(args.res_path+batch['folder'][i]+'ADC/')
            if not os.path.exists(args.res_path+batch['folder'][i]+'ADC_/'):
                os.mkdir(args.res_path+batch['folder'][i]+'ADC_/')
            cv2.imwrite(args.res_path+batch['folder'][i]+'T2/'+batch['im_num'][i]+'.png',(src[i,:,:,0]+1)/2*255)
            cv2.imwrite(args.res_path+batch['folder'][i]+'ADC/'+batch['im_num'][i]+'.png',(tgt[i,:,:,0]+1)/2*255)
            cv2.imwrite(args.res_path+batch['folder'][i]+'T2_/'+batch['im_num'][i]+'.png',(tgt2src[i,:,:,0]+1)/2*255)
            cv2.imwrite(args.res_path+batch['folder'][i]+'ADC_/'+batch['im_num'][i]+'.png',(src2tgt[i,:,:,0]+1)/2*255)









def get_data_loaders(args):
    val_dataloader=ImageQualityPair(root=args.data_path,mode='score/')
    val_loader=torch.utils.data.DataLoader(val_dataloader,batch_size=args.batch_size,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=False)

    return val_loader


if __name__ == '__main__':
    parser=TestArgParser()
    get_images(parser.parse_args())