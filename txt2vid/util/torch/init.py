import torch.nn as nn
import torch.nn.init as tinit

def _weight_init(layer, init_func=None):
    name = layer.__class__.__name__
    if 'Linear' in name or 'Conv' in name or 'Embedding' in name:
        if hasattr(layer, 'weight') and layer.weight is not None:
            if hasattr(layer, 'is_residual') and layer.is_residual:
                import math
                factor = math.sqrt(2)
                init_func(layer.weight, gain=factor)
            else:
                init_func(layer.weight)


        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data.fill_(0.0)

    elif 'BatchNorm' in name:
        if hasattr(layer, 'weight') and layer.weight is not None:
            layer.weight.data.fill_(1.0)

        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init(model, init_method=None):
    from functools import partial

    init_func = None
    if init_method == 'xavier':
        init_func = tinit.xavier_normal_
    elif init_method == 'ortho':
        init_func = tinit.orthogonal_
    elif init_method == 'normal':
        init_func = partial(tinit.normal_, mean=0, std=0.02)
    else:
        assert(False)

    model.apply(partial(_weight_init, init_func=init_func))
