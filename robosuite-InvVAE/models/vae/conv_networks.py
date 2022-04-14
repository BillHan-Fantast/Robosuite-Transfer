import torch
from torch import nn as nn

from rlkit.pythonplusplus import identity
from models.vae.basic_networks import ConvTranspose2dBlock, Conv2dBlock, ResBlock
import numpy as np


class TwoHeadDCNN(nn.Module):
    def __init__(
            self,
            tie_fc_layers,
            **kwargs
    ):
        super().__init__()

        self.decoders = nn.ModuleDict({
            'source': DCNN(**kwargs),
            'target': DCNN(**kwargs)
        })

        if tie_fc_layers:
            self.decoders['source'].tie_fc_layers(self.decoders['target'].fc_layers,
                                                  self.decoders['target'].last_fc, True)

    def forward(self, input, domain):
        return self.decoders[domain].forward(input)


class TwoBiasDCNN(nn.Module):
    def __init__(
            self,
            tie_deconv_bias,
            **kwargs
    ):
        super().__init__()

        self.decoders = nn.ModuleDict({
            'source': DCNN(**kwargs),
            'target': DCNN(**kwargs)
        })

        self.decoders['source'].tie_fc_layers(self.decoders['target'].fc_layers,
                                              self.decoders['target'].last_fc, False)
        self.decoders['source'].tie_deconv_layers(self.decoders['target'].deconv_layers,
                                                  self.decoders['target'].deconv_output, tie_deconv_bias)

    def forward(self, input, domain):
        return self.decoders[domain].forward(input)


class DCNN(nn.Module):
    def __init__(
            self,
            fc_input_size,
            hidden_sizes,

            deconv_input_width,
            deconv_input_height,
            deconv_input_channels,

            deconv_output_kernel_size,
            deconv_output_strides,
            deconv_output_channels,

            kernel_sizes,
            n_channels,
            strides,
            paddings,
            activations,
            normalizations,

            hidden_activation=nn.ReLU(),
            output_activation=identity,
            **kwargs
    ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings) == \
               len(activations) == \
               len(normalizations)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_channels * self.deconv_input_height * self.deconv_input_width

        self.deconv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)
            self.fc_layers.append(fc_layer)
            fc_input_size = hidden_size
        self.last_fc = nn.Linear(fc_input_size, deconv_input_size)

        for out_channels, kernel_size, stride, padding, activation, normalization in \
                zip(n_channels, kernel_sizes, strides, paddings, activations, normalizations):
            deconv = ConvTranspose2dBlock(deconv_input_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          activation=activation,
                                          normalization=normalization)

            deconv_layer = deconv
            self.deconv_layers.append(deconv_layer)
            deconv_input_channels = out_channels

        self.deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
            padding=0,
        )

    def forward(self, input):
        h = input
        for layer in self.fc_layers:
            h = self.hidden_activation(layer(h))
        h = self.hidden_activation(self.last_fc(h))
        h = h.reshape(-1, self.deconv_input_channels, self.deconv_input_height, self.deconv_input_width)
        h = self.conv_forward(h)
        output = self.output_activation(self.deconv_output(h))
        return output

    def conv_forward(self, input):
        h = input
        for layer in self.deconv_layers:
            h = layer(h)
        return h

    def tie_fc_layers(self, fc_layers, last_fc, tie_bias=False):
        assert len(fc_layers) == len(self.fc_layers)
        for idx in range(len(fc_layers)):
            tie_networks(self.fc_layers[idx], fc_layers[idx], tie_bias)

        tie_networks(self.last_fc, last_fc, tie_bias)

    def tie_deconv_layers(self, deconv_layers, deconv_output, tie_bias=False):
        assert len(deconv_layers) == len(self.deconv_layers)
        for idx in range(len(deconv_layers)):
            self.deconv_layers[idx].set_weight(deconv_layers[idx].weight)
            if tie_bias:
                self.deconv_layers[idx].set_bias(deconv_layers[idx].bias)

        tie_networks(self.deconv_output, deconv_output, tie_bias)


class CNN(nn.Module):
    def __init__(
            self,
            image_shape,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            activations,
            normalizations,
            num_residual,
            hidden_sizes=None,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings) == \
               len(activations) == \
               len(normalizations)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = image_shape[1]
        self.input_height = image_shape[0]
        self.input_channels = image_shape[2]
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        input_channels = self.input_channels
        conv_shape = image_shape[:-1]
        for out_channels, kernel_size, stride, padding, activation, normalization in \
                zip(n_channels, kernel_sizes, strides, paddings, activations, normalizations):
            conv = Conv2dBlock(input_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               activation=activation,
                               normalization=normalization)
            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels
            conv_shape = conv_out_shape(conv_shape, padding, kernel_size, stride)

        for _ in range(num_residual):
            self.conv_layers.append(
                ResBlock(input_channels, 'in', 'relu')
            )

        fc_input_size = int(np.prod(conv_shape) * input_channels)
        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)
            self.fc_layers.append(fc_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, output_size)

    def forward(self, input, domain):
        h = self.conv_forward(input, domain)
        h = h.reshape(h.shape[0], -1)

        for layer in self.fc_layers:
            h = self.hidden_activation(layer(h))
        output = self.output_activation(self.last_fc(h))
        return output

    def conv_forward(self, input, domain):
        h = input
        for layer in self.conv_layers:
            h = layer(h, domain)
        return h


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def tie_networks(src, trg, tie_bias=False):
    assert type(src) == type(trg)
    src.weight = trg.weight
    if tie_bias:
        src.bias = trg.bias
