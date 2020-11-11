# coding=utf-8

import math
import torch

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_

def init_method_uniform(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.uniform_(tensor, a=-sigma, b=sigma)
    return init_

def init_method_xavier_uniform(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.xavier_uniform_(tensor)
    return init_

def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def get_linear_layer(rows, columns, init_method):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


def get_params_for_weight_decay_optimization(module):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        weight_decay_params['params'].extend(
            [p for n, p in list(module_._parameters.items())
                if p is not None and n != 'bias'])
        no_weight_decay_params['params'].extend(
            [p for n, p in list(module_._parameters.items())
                if p is not None and n == 'bias'])
    return weight_decay_params, no_weight_decay_params