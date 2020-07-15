"""
BSD 3-Clause License

Copyright (c) 2018, NVIDIA Corporation
All rights reserved.
"""
from torch import nn
from torch import max as torchmax
from math import sqrt

from tacotron2_model.attention import Attention
from tacotron2_model.decoder import Decoder
from tacotron2_model.encoder import Encoder
from tacotron2_model.layers import LocationLayer
from tacotron2_model.normalizers import ConvNorm, LinearNorm
from tacotron2_model.postnet import Postnet
from tacotron2_model.prenet import Prenet
from tacotron2_model.utils import to_gpu, get_mask_from_lengths


class Tacotron2(nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_symbols,
        symbols_embedding_dim,
        encoder_n_convolutions,
        encoder_embedding_dim,
        encoder_kernel_size,
        attention_rnn_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
        decoder_rnn_dim,
        prenet_dim,
        max_decoder_steps,
        gate_threshold,
        p_attention_dropout,
        p_decoder_dropout,
        postnet_embedding_dim,
        postnet_kernel_size,
        postnet_n_convolutions,
    ):
        super(Tacotron2, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = 1
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size)
        self.decoder = Decoder(
            n_mel_channels,
            self.n_frames_per_step,
            encoder_embedding_dim,
            attention_rnn_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            decoder_rnn_dim,
            prenet_dim,
            max_decoder_steps,
            gate_threshold,
            p_attention_dropout,
            p_decoder_dropout,
        )
        self.postnet = Postnet(n_mel_channels, postnet_embedding_dim, postnet_n_convolutions, postnet_kernel_size)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torchmax(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return ((text_padded, input_lengths, mel_padded, max_len, output_lengths), (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
