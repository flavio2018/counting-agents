# from: https://discuss.pytorch.org/t/passing-hidden-layers-to-convlstm/52814/3

import torch
from torch import nn

def initialize_weights(self, layer):
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

class HadamardProduct(nn.Module):
    """A Hadamard product layer.
    
    Args:
        shape: The shape of the layer."""
       
    def __init__(self, shape):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(*shape))
        self.bias = nn.Parameter(torch.empty(*shape))
           
    def forward(self, x):
        return x * self.weights

    
class ConvLSTMCell(BaseModule):
    """A convolutional LSTM cell.

    Implementation details follow closely the following paper:

    Shi et al. -'Convolutional LSTM Network: A Machine Learning 
    Approach for Precipitation Nowcasting' (2015).
    Accessible at https://arxiv.org/abs/1506.04214

    The parameter names are drawn from the paper's Eq. 3.

    Args:
        input_bands: The number of bands in the input data.
        input_dim: The length of of side of input data. Data is
            presumed to have identical width and heigth."""

    def __init__(self, input_bands, input_dim,  kernels, dropout, batch_norm):
        super().__init__()

        self.input_bands = input_bands
        self.input_dim = input_dim
        self.kernels = kernels
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.kernel_size = 3
        self.padding = 1  # Preserve dimensions

        self.input_conv_params = {
            'in_channels': self.input_bands,
            'out_channels': self.kernels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'bias': True
        }

        self.hidden_conv_params = {
            'in_channels': self.kernels,
            'out_channels': self.kernels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'bias': True
        }

        self.state_shape = (
            1,
            self.kernels,
            self.input_dim,
            self.input_dim
        )

        self.batch_norm_layer = None
        if self.batch_norm:
            self.batch_norm_layer = nn.BatchNorm2d(num_features=self.input_bands)

        # Input Gates
        self.W_xi = nn.Conv2d(**self.input_conv_params)
        self.W_hi = nn.Conv2d(**self.hidden_conv_params)
        self.W_ci = HadamardProduct(self.state_shape)

        # Forget Gates
        self.W_xf = nn.Conv2d(**self.input_conv_params)
        self.W_hf = nn.Conv2d(**self.hidden_conv_params)
        self.W_cf = HadamardProduct(self.state_shape)

        # Memory Gates
        self.W_xc = nn.Conv2d(**self.input_conv_params)
        self.W_hc = nn.Conv2d(**self.hidden_conv_params)

        # Output Gates
        self.W_xo = nn.Conv2d(**self.input_conv_params)
        self.W_ho = nn.Conv2d(**self.hidden_conv_params)
        self.W_co = HadamardProduct(self.state_shape)

        # Dropouts
        self.H_drop = nn.Dropout2d(p=self.dropout)
        self.C_drop = nn.Dropout2d(p=self.dropout)

        self.apply(initialize_weights) 
