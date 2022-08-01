import argparse
import os
import json
import torch

from util import args_to_list, gan_class_name, print_err, str_to_bool


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', type=gan_class_name, required=True,
                                 choices=('CycleGAN', 'CycleFlow', 'Flow2Flow'),
                                 help='Name of model to run. Case-insensitive.')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
        self.parser.add_argument('--checkpoints_dir', type=str, default='ckpts/',
                                 help='Directory in which to save checkpoints.')
        self.parser.add_argument('--ckpt_path', type=str, default='',
                                 help='Path to model checkpoint to load.')
        self.parser.add_argument('--data_path',type=str,default='../data/ProstateImageQuality1/',
                                 help='data path')
        self.parser.add_argument('--direction', type=lambda s: s.lower(), default='ab', choices=('ab', 'ba'),
                                 help='Direction of source to target mapping.')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                                 help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--initializer', type=str, default='normal', choices=('xavier', 'he', 'normal'),
                                 help='Initializer to use for all network parameters.')
        self.parser.add_argument('--kernel_size_d', default=4, type=int, help='Size of the discriminator\'s kernels.')
        self.parser.add_argument('--name', type=str, required=True, help='Experiment name.')
        self.parser.add_argument('--norm_type', type=str, default='instance', choices=('instance', 'batch', 'group'),
                                 help='Normalization type.')
        self.parser.add_argument('--num_scales', default=2, type=int, help='Number of scales in Real NVP.')
        self.parser.add_argument('--num_blocks', default=8, type=int, help='Number of res. blocks in s/t (Real NVP).')
        self.parser.add_argument('--resize_shape', type=str, default='286,286',
                                 help='Comma-separated 2D shape for images after resizing (before cropping).\
                                      By default, no resizing is applied.')
        self.parser.add_argument('--crop_shape', type=str, default='256,256',
                                 help='Comma-separated 2D shape for images after cropping (crop comes after resize).\
                                      By default, no cropping is applied.')
        self.parser.add_argument('--num_channels', default=1, type=int, help='Number of channels in an image.')
        self.parser.add_argument('--num_channels_d', default=64, type=int,
                                 help='Number of filters in the discriminator\'s first convolutional layer.')
        self.parser.add_argument('--num_channels_g', default=64, type=int,
                                 help='Number of filters in the generator\'s first convolutional layer.')
        self.parser.add_argument('--num_workers', default=16, type=int, help='Number of DataLoader worker threads.')
        self.parser.add_argument('--phase', default='train', type=str, help='One of "train", "val", or "test".')
        self.parser.add_argument('--use_dropout', default=False, type=str_to_bool,
                                 help='Use dropout in the generator.')
        self.is_training = None
        self.no_mixer = None
        self.use_dropout = None

    def parse_args(self):
        args = self.parser.parse_args()

        # Add configuration flags outside of the CLI
        args.is_training = self.is_training
        #if not args.is_training and not args.ckpt_path:
        #    raise ValueError('Must specify --ckpt_path in test mode.')

        # Set up resize and crop
        args.resize_shape = args_to_list(args.resize_shape, arg_type=int, allow_empty=False)
        args.crop_shape = args_to_list(args.crop_shape, arg_type=int, allow_empty=False)

        # Set up available GPUs
        args.gpu_ids = args_to_list(args.gpu_ids, arg_type=int, allow_empty=True)
        if len(args.gpu_ids) > 0:
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda:{}'.format(args.gpu_ids[0])
        else:
            args.device = 'cpu'

        # Set up save dir and output dir (test mode only)
        args.save_dir = os.path.join(args.checkpoints_dir, args.name)
        os.makedirs(args.save_dir, exist_ok=True)
        if self.is_training:
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as fh:
                json.dump(vars(args), fh, indent=4, sort_keys=True)
                fh.write('\n')
        else:
            args.results_dir = os.path.join(args.res_path, args.name)
            os.makedirs(args.res_path, exist_ok=True)

        return args



class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        self.parser.add_argument('--iters_per_visual', type=int, default=3200,
                                 help='Number of iterations between visualizing training examples.')
        self.parser.add_argument('--iters_per_print', type=int, default=320,
                                 help='Number of iterations between printing loss to the console and TensorBoard.')
        self.parser.add_argument('--iters_per_eval', type=int, default=1600,
                                 help='Number of iterations between each model eval/save.')
        self.parser.add_argument('--max_ckpts', type=int, default=20,
                                 help='Maximum number of checkpoints to save.')
        self.parser.add_argument('--metric_name', type=str, default='MSE_src2tgt',
                                 help='Name of metric to determine best checkpoint when saving.')
        self.parser.add_argument('--maximize_metric', type=str_to_bool, default=False,
                                 help='Whether to maximize metric `metric_name` when saving.')
        self.parser.add_argument('--lambda_src', type=float, default=10., help='Source image cycle loss weight.')
        self.parser.add_argument('--lambda_tgt', type=float, default=10., help='Target image cycle loss weight.')
        self.parser.add_argument('--lambda_id', type=float, default=0.5, help='Ratio of loss weights ID:GAN.')
        self.parser.add_argument('--lambda_l1', type=float, default=100., help='Ratio of loss weights L1:GAN.')
        self.parser.add_argument('--lambda_mle', type=float, default=1e-4, help='Ratio of loss weights MLE:GAN.')
        self.parser.add_argument('--lambda_lat',type=float,default=0,help='Ratio of loss weights latent:GAN.')
        self.parser.add_argument('--beta_1', type=float, default=0.5, help='Adam hyperparameter: beta_1.')
        self.parser.add_argument('--beta_2', type=float, default=0.999, help='Adam hyperparameter: beta_2.')
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Adam hyperparameter: initial learning rate.')
        self.parser.add_argument('--rnvp_beta_1', type=float, default=0.5, help='RealNVP Adam hyperparameter: beta_1.')
        self.parser.add_argument('--rnvp_beta_2', type=float, default=0.999, help='RealNVP Adam hyperparameter: beta_2.')
        self.parser.add_argument('--rnvp_lr', type=float, default=2e-4, help='RealNVP learning rate.')
        self.parser.add_argument('--weight_norm_l2', type=float, default=5e-5,
                                 help='L2 regularization factor for weight norm scale factors.')
        self.parser.add_argument('--lr_policy', type=str, default='linear',
                                 help='Learning rate schedule policy. See modules/optim.py for details.',
                                 choices=('linear', 'plateau', 'step'))
        self.parser.add_argument('--lr_step_epochs', type=int, default=25,
                                 help='Number of epochs between each divide-by-10 step (step policy only).')
        self.parser.add_argument('--lr_warmup_epochs', type=int, default=25,
                                 help='Number of epochs before we start decaying the learning rate (linear only).')
        self.parser.add_argument('--lr_decay_epochs', type=int, default=25,
                                 help='Number of epochs to decay the learning rate linearly to 0 (linear only).')
        self.parser.add_argument('--use_mixer', default=True, type=str_to_bool,
                                 help='Use image buffer during training. \
                                      Note that mixer is disabled for conditional GAN by default.')
        self.parser.add_argument('--num_epochs', type=int, default=50,
                                 help='Number of epochs to train. If 0, train forever.')
        self.parser.add_argument('--num_visuals', type=int, default=4, help='Maximum number of visuals per batch.')
        self.parser.add_argument('--clip_gradient', type=float, default=0.,
                                 help='Maximum gradient norm. Setting to 0 disables gradient clipping.')
        self.parser.add_argument('--clamp_jacobian', type=str_to_bool, default=False,
                                 help='Use Jacobian Clamping from https://arxiv.org/abs/1802.08768.')
        self.parser.add_argument('--jc_lambda_min', type=float, default=1.,
                                 help='Jacobian Clamping lambda_min parameter.')
        self.parser.add_argument('--jc_lambda_max', type=float, default=20.,
                                 help='Jacobian Clamping lambda_max parameter.')



class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--num_examples', type=int, default=-1, help='Number of examples.')
        self.parser.add_argument('--res_path', type=str, default='results/', help='Save dir for test results.')
        self.parser.add_argument('--model_path',type=str,default='ckpts/test2/best.pth.tar',help='model path')