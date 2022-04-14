import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, dim, normalization='in', activation='relu', padding_mode='zeros'):
        super().__init__()

        self.network = nn.ModuleList()
        self.network.append(Conv2dBlock(dim, dim, 3, 1, 1, normalization=normalization,
                                        activation=activation, padding_mode=padding_mode))
        self.network.append(Conv2dBlock(dim, dim, 3, 1, 1, normalization=normalization,
                                        activation='none', padding_mode=padding_mode))

    def forward(self, x, domain):
        residual = x
        for layer in self.network:
            x = layer(x, domain)
        x += residual
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 normalization='none', activation='relu', padding_mode='zeros'):
        super().__init__()

        normalization_dim = output_dim
        if normalization == 'bn':
            self.normalization = nn.BatchNorm2d(normalization_dim)
        elif normalization == 'adabn':
            self.normalization = nn.ModuleDict({
                'source': nn.BatchNorm2d(normalization_dim),
                'target': nn.BatchNorm2d(normalization_dim)
            })
        elif normalization == 'in':
            self.normalization = nn.InstanceNorm2d(normalization_dim)
        elif normalization == 'none':
            self.normalization = None
        else:
            assert 0, "Unsupported normalization: {}".format(normalization)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                              padding=padding, padding_mode=padding_mode)

    def forward(self, x, domain='source'):
        x = self.conv(x)
        if isinstance(self.normalization, nn.ModuleDict):
            x = self.normalization[domain](x)
        elif self.normalization:
            x = self.normalization(x)
        if self.activation:
            x = self.activation(x)
        return x


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 normalization='none', activation='relu'):
        super().__init__()

        normalization_dim = output_dim
        if normalization == 'bn':
            self.normalization = nn.BatchNorm2d(normalization_dim)
        elif normalization == 'in':
            self.normalization = nn.InstanceNorm2d(normalization_dim)
        elif normalization == 'none':
            self.normalization = None
        else:
            assert 0, "Unsupported normalization: {}".format(normalization)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.deconv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding)

    def forward(self, x):
        x = self.deconv(x)
        if self.normalization:
            x = self.normalization(x)
        if self.activation:
            x = self.activation(x)
        return x

    @property
    def weight(self):
        return self.deconv.weight

    @property
    def bias(self):
        return self.deconv.bias

    def set_weight(self, weight):
        self.deconv.weight = weight

    def set_bias(self, bias):
        self.deconv.bias = bias


class ForwardModel(nn.Module):
    def __init__(self, state_dim, object_dim, action_dim, hidden_units):
        super().__init__()
        self.network = MLP_Layers(state_dim + object_dim + action_dim,
                                  state_dim + object_dim, hidden_units)

        self.state_dim = state_dim
        self.object_dim = object_dim

    def forward(self, st, ot, ac):
        delta = self.network(torch.cat([st, ot, ac], dim=-1))
        delta_st = delta[..., :self.state_dim]
        delta_ot = delta[..., self.state_dim:]
        return delta_st + st, delta_ot + ot

    def get_norm(self):
        with torch.no_grad():
            norm = 0.
            for net in list(self.network.parameters()):
                norm += torch.sum(net ** 2)
        return torch.sqrt(norm)


class InverseModel(nn.Module):
    def __init__(self, state_dim, object_dim, action_dim, hidden_units):
        super().__init__()
        self.network = MLP_Layers(2 * (state_dim + object_dim), action_dim, hidden_units)

    def forward(self, st, ot, n_st, n_ot):
        inputs = torch.cat([st, ot, n_st, n_ot], dim=-1)
        outputs = self.network(inputs)
        return torch.tanh(outputs) * 1.05

    def get_norm(self):
        with torch.no_grad():
            norm = 0.
            for net in list(self.network.parameters()):
                norm += torch.sum(net ** 2)
        return torch.sqrt(norm)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        self.network = MLP_Layers(input_dim, 2, hidden_units,
                                  output_activation=nn.LogSoftmax(dim=-1))

    def forward(self, x):
        return self.network(x)


def MLP_Layers(input_dim, output_dim, hidden_units, hidden_activation=nn.ELU(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)