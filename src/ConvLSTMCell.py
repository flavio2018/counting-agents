"""
This script contains the PyTorch implementation of the core ConvLSTM cell of the agent model.

References:
  - https://discuss.pytorch.org/t/passing-hidden-layers-to-convlstm/52814/3
"""

import torch
from torch import nn
from HadamardProduct import HadamardProduct
from utils import initialize_weights
    
class ConvLSTMCell(nn.Module):
    """A convolutional LSTM cell.

    Implementation details follow closely the following paper:

    Shi et al. -'Convolutional LSTM Network: A Machine Learning 
    Approach for Precipitation Nowcasting' (2015).
    Accessible at https://arxiv.org/abs/1506.04214

    The parameter names are drawn from the paper's Eq. 3.

    Args:
        input_channels: The number of channels in the input data.
        input_dim: The length of of side of input data. Data is
            presumed to have identical width and heigth.
        n_kernels: The number of kernels of Conv2d layers.
        dropout: The value of p parameter for dropout layers.
        batch_norm: Boolean value to indicate whether to use a batch normalization
            layer"""

    def __init__(self, input_channels, input_dim,  n_kernels, dropout, batch_norm):
        super().__init__()

        self.input_channels = input_channels
        self.input_dim = input_dim
        self.n_kernels = n_kernels
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.kernel_size = 3
        self.padding = 1  # Preserve dimensions

        self.input_conv_params = {
            'in_channels': self.input_channels,
            'out_channels': self.n_kernels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'bias': True
        }

        self.hidden_conv_params = {
            'in_channels': self.n_kernels,
            'out_channels': self.n_kernels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'bias': True
        }

        self.state_shape = (
            1, # no batches
            self.n_kernels,
            self.input_dim,
            self.input_dim
        )
        
        self.H = torch.zeros(self.state_shape)
        self.C = torch.zeros(self.state_shape)
        
        self.batch_norm_layer = None
        if self.batch_norm:
            self.batch_norm_layer = nn.BatchNorm2d(num_features=self.input_channels)

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
        
        self.b_i = torch.zeros(self.state_shape, requires_grad=True)
        self.b_f = torch.zeros(self.state_shape, requires_grad=True)
        self.b_c = torch.zeros(self.state_shape, requires_grad=True)
        self.b_o = torch.zeros(self.state_shape, requires_grad=True)
        
        self.apply(initialize_weights)
    
    def forward(self, x):
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        
        i_t = sigmoid(self.W_xi(x) + self.W_hi(self.H) + self.W_ci(self.C) + self.b_i)
        f_t = sigmoid(self.W_xf(x) + self.W_hf(self.H) + self.W_cf(self.C) + self.b_f)
        C_t = f_t * self.C + i_t * tanh(self.W_xc(x) + self.W_hc(self.H) + self.b_c)
        C_t = self.C_drop(C_t)
        o_t = sigmoid(self.W_xo(x) + self.W_ho(self.H) + self.W_co(self.C) + self.b_o)
        H_t = o_t * tanh(C_t)
        H_t = self.H_drop(H_t)
        
        self.C = C_t
        self.H = H_t
        
        return H_t, C_t
