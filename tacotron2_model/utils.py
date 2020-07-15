"""
BSD 3-Clause License

Copyright (c) 2018, NVIDIA Corporation
All rights reserved.
"""


def to_gpu(x):
    x = x.contiguous()
    x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask
