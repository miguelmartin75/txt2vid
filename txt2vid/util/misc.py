import numpy as np

def gen_perm(n):
    old_perm = np.array(range(n))
    new_perm = np.random.permutation(old_perm)
    while (new_perm == old_perm).all():
        new_perm = np.random.permutation(old_perm)
    return new_perm

def count_params(model):
    return sum(p.numel() for p in model.parameters())

