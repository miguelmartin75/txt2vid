import numpy as np
import random
import torch

from txt2vid.util.log import warn, status

def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 100000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

def set_cuda(use_cuda=False):
    import torch
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    if torch.cuda.is_available() and not use_cuda:
        warn('cuda is available')

    return torch.device("cuda:0" if use_cuda else "cpu")

def setup(args):
    seed = set_seed(args.seed)
    device = set_cuda(use_cuda=args.cuda)
    status('Seed: %d' % seed)
    status('Device set to: %s' % device)
    return seed, device
