import time
import torch

from tqdm import tqdm
from util import AverageMeter


def evaluate(model, data_loader, criteria):
    """Evaluate a model.
    Args:
        model (torch.nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Loader for data to evaluate on.
        criteria (dict): Dictionary mapping strings to functions, where each
            criterion function takes outputs and targets, and produces a number.
        max_examples (int): Maximum number of examples on which to evaluate.
        batch_hook (func): Callback to call with (src, src2tgt, tgt, tgt2src) images
            after each batch.
    Returns:
        Dictionary mapping strings (one per criterion) to the average value
            returned by that criterion on the dataset.
    """
    time_meter = AverageMeter()
    meters = {k: AverageMeter() for k in criteria}

    with torch.no_grad():
        for batch in data_loader:
            start = time.time()
            batch_size = batch['t2'].size(0)

            # Evaluate one src -> tgt batch
            model.set_inputs(batch['t2'], batch['adc'])
            model.test()
            for criterion_name, criterion_fn in criteria.items():
                if criterion_name.endswith('tgt2src'):
                    criterion_val = criterion_fn(model.tgt2src, model.src).item()
                else:
                    # Assume forward direction
                    criterion_val = criterion_fn(model.src2tgt, model.tgt).item()
                criterion_meter = meters[criterion_name]
                criterion_meter.update(criterion_val, batch_size)
            time_meter.update(time.time() - start, batch_size)



    stats = {k: v.avg for k, v in meters.items()}

    return stats