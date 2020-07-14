"""
BSD 3-Clause License

Copyright (c) 2018, NVIDIA Corporation
All rights reserved.
"""
from torch import nn

from normalizers import ConvNorm, LinearNorm

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2, attention_n_filters, kernel_size=attention_kernel_size, padding=padding, bias=False, stride=1, dilation=1
        )
        self.location_dense = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain="tanh")

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention
