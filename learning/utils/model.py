from torch import nn


def enable_dropout(model: nn.Module):
    """Enable any dropout layer"""
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()