"""
BSD 3-Clause License

Copyright (c) 2018, NVIDIA Corporation
All rights reserved.
"""
from torch import nn

from tacotron2_model.normalizers import ConvNorm


class Encoder(nn.Module):
    """
    Encoder module:
    - Three 1-d convolution banks
    - Bidirectional LSTM
    """

    def __init__(self, encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_embedding_dim, int(encoder_embedding_dim / 2), 1, batch_first=True, bidirectional=True
        )

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs
