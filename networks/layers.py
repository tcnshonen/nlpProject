import torch
import torch.nn as nn


def activation(name):
    if name == 'elu':
        act = nn.elu()
    elif name == 'leakyrelu':
        act = nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        act = nn.Tanh()
    elif name == 'sigmoid':
        act = nn.Sigmoid()
    elif name =='relu':
        act = nn.ReLU()
    else:
        act = None
    return act


class ConvLayer(nn.Module):
    def __init__(self, in_num, out_num, kernel=1, stride=1, padding=0, output_padding=0, norm=True, dropout=True, activation_name='relu', transpose=False):
        super(ConvLayer, self).__init__()
        self.norm = norm
        self.dropout = dropout
        self.activation_name = activation_name
        self.transpose = transpose
        self.config = {'kernel_size': kernel, 'stride': stride, 'padding': padding}

        modules = []
        if self.transpose:
            self.config['output_padding'] = output_padding
            modules.append(nn.ConvTranspose2d(in_num, out_num, **self.config))
        else:
            modules.append(nn.Conv2d(in_num, out_num, **self.config))

        if self.norm:
            modules.append(nn.BatchNorm2d(out_num))

        modules.append(activation(activation_name))

        if self.dropout:
            modules.append(nn.Dropout2d(p=0.2))

        self.layers = nn.Sequential(*modules)


    def forward(self, x):
        x = self.layers(x)
        return x


class LinearLayer(nn.Module):
    def __init__(self, in_num, out_num, norm=True, dropout=True, activation_name='relu'):
        super(LinearLayer, self).__init__()
        self.norm = norm
        self.dropout = dropout
        self.activation_name = activation_name

        modules = [nn.Linear(in_num, out_num)]

        if self.norm:
            modules.append(nn.BatchNorm1d(out_num))

        modules.append(activation(activation_name))

        if self.dropout:
            modules.append(nn.Dropout(p=0.5))

        self.layers = nn.Sequential(*modules)


    def forward(self, x):
        x = self.layers(x)
        return x
