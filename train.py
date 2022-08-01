import models

from args import TrainArgParser
from dataloader import ImageQualityPair
from evaluation import evaluate

from logger import TrainLogger
from saver import ModelSaver
import torch.utils.data
import torch.nn.functional as F
import time

import warnings
warnings.filterwarnings("ignore")


def train(args):
    # Get model
    model = models.__dict__[args.model](args)
    if args.ckpt_path:
        model = ModelSaver.load_model(model, args.ckpt_path, args.gpu_ids, is_training=True)
    model = model.to(args.device)
    model.train()

    # Get loader, logger, and saver
    train_loader, val_loader = get_data_loaders(args)
    logger = TrainLogger(args, model, dataset_len=len(train_loader.dataset))
    saver = ModelSaver(args.save_dir, args.max_ckpts, metric_name=args.metric_name,
                       maximize_metric=args.maximize_metric, keep_topk=True)

    # Train
    while not logger.is_finished_training():
        logger.start_epoch()
        for _,batch in enumerate(train_loader):
            t=time.time()
            logger.start_iter()

            # Train over one batch
            model.set_inputs(batch['t2'], batch['adc'])
            model.train_iter()

            logger.end_iter()

            # Evaluate
            if logger.global_step % args.iters_per_eval < args.batch_size:
                t=time.time()
                criteria = {'MSE_src2tgt': F.mse_loss, 'MSE_tgt2src': F.mse_loss}
                stats = evaluate(model, val_loader, criteria)
                logger.log_scalars({'val_' + k: v for k, v in stats.items()})
                saver.save(logger.global_step, model,
                           stats[args.metric_name], args.device)
        logger.end_epoch()


def get_data_loaders(args):
    train_dataloader=ImageQualityPair(root=args.data_path,mode='no_score/')
    train_loader=torch.utils.data.DataLoader(train_dataloader,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,sampler=None,drop_last=True)
    val_dataloader=ImageQualityPair(root=args.data_path,mode='score/')
    val_loader=torch.utils.data.DataLoader(val_dataloader,batch_size=args.batch_size,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=False)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())