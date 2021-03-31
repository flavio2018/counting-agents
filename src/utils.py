"""
This file contains some utility functions used in different classes.
"""

import torch
from torch import nn

def initialize_weights(layer):
    """Initialize a layer's weights and biases.

    Args:
        layer: A PyTorch Module's layer."""
    if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        pass
    else:
        try:
            nn.init.xavier_normal_(layer.weight)
        except AttributeError:
            pass
        try:
            nn.init.uniform_(layer.bias)
        except (ValueError, AttributeError):
            pass 
