import os
from datetime import datetime
from tensorboardX import SummaryWriter
import util
from time import time


class BaseLogger(object):
    def __init__(self, args, model, dataset_len):

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.dataset_len = dataset_len
        self.save_dir = args.save_dir if args.is_training else args.results_dir
        self.log_path = os.path.join(self.save_dir, '{}.log'.format(args.name))
        self.model = model
        log_dir = os.path.join('logs', args.name + '_' + datetime.now().strftime('%b%d_%H%M'))
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 1
        # Current iteration in epoch (i.e., # examples seen in the current epoch)
        self.iter = 0
        # Current iteration overall (i.e., total # of examples seen)
        self.global_step = round_down((self.epoch - 1) * dataset_len, args.batch_size)
        self.iter_start_time = None
        self.epoch_start_time = None

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def log_scalars(self, scalar_dict, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {:.3g}]'.format(k, v))
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, self.global_step)

    def start_iter(self, src_filenames=None):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError



class TrainLogger(BaseLogger):
    """Logs training info."""
    def __init__(self, args, model, dataset_len):
        super(TrainLogger, self).__init__(args, model, dataset_len)

        assert args.is_training
        assert args.iters_per_print % args.batch_size == 0, \
            'iters_per_print must be divisible by batch_size'
        assert args.iters_per_visual % args.batch_size == 0, \
            'iters_per_visual must be divisible by batch_size'

        self.batch_size = args.batch_size
        self.iters_per_print = args.iters_per_print
        self.iters_per_visual = args.iters_per_visual
        self.num_visuals = args.num_visuals
        self.num_epochs = args.num_epochs

    def start_iter(self, src_filenames=None):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

        # Periodically write to the log and TensorBoard
        if self.global_step % self.iters_per_print == 0:
            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}] '\
                      .format(self.epoch, self.iter, self.dataset_len, avg_time)

            # Write the current error report
            loss_dict = self.model.get_loss_dict()
            loss_keys = ['loss_g', 'loss_d']  # Can add other losses here (e.g., 'loss_g_l1').
            loss_strings = ['{}: {:.3g}'.format(k, loss_dict[k]) for k in loss_keys]
            message += ', '.join(loss_strings)

            # Write all errors as scalars to the graph
            for k, v in loss_dict.items():
                # Group generator and discriminator losses
                if k.startswith('loss_d'):
                    k = 'd/' + k
                else:
                    k = 'g/' + k
                self.summary_writer.add_scalar(k, v, self.global_step)

            self.write(message)

        # Periodically visualize training examples
        if self.global_step % self.iters_per_visual == 0:
            image_dict = self.model.get_image_dict()
            if len(image_dict) % 3 == 0:
                # Image dict contains full cycle (e.g., in CycleGAN)
                plot_groups = ('src', 'tgt')
                image_keys = {'src': ('src', 'src2tgt', 'src2tgt2src'),
                              'tgt': ('tgt', 'tgt2src', 'tgt2src2tgt')}
                nrow = 3
            elif len(image_dict) > 2:
                # Image dict contains src2tgt, tgt2src (e.g., in CycleFlow/Flow2Flow)
                plot_groups = ('src', 'tgt')
                image_keys = {'src': ('src', 'src2tgt'),
                              'tgt': ('tgt', 'tgt2src')}
                nrow = 2
            else:
                # Image dict just contains src -> tgt (e.g., in Pix2Pix)
                plot_groups = ('example',)
                image_keys = {'example': ('src', 'src2tgt')}
                nrow = 2

            # Concatenate images into triples
            for plot_group in plot_groups:
                image_batches = [image_dict[k] for k in image_keys[plot_group]]
                for i in range(self.num_visuals):
                    images = []
                    for image_batch in image_batches:
                        if len(image_batch) > i:
                            images.append(image_batch[i])
                    if len(images) == 0:
                        continue
                    images_title = '{}/{}_{}'.format(plot_group, '_'.join(image_keys[plot_group]), i+1)
                    images_concat = util.make_grid(images, nrow=nrow, padding=4, pad_value=(243, 124, 42))
                    self.summary_writer.add_image(images_title, images_concat, self.global_step)

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))

    def end_epoch(self):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        self.write('[end of epoch {}, epoch time: {:.2f}]'.format(self.epoch, time() - self.epoch_start_time))

        # Update the learning rate according to the LR schedulers
        self.model.on_epoch_end()
        learning_rate = self.model.get_learning_rate()
        self.summary_writer.add_scalar('hpm/lr', learning_rate, self.global_step)

        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch