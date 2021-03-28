# from: https://discuss.pytorch.org/t/passing-hidden-layers-to-convlstm/52814/3

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

    
class ConvLSTMCell(nn.Module):
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
        
        self.H = torch.zeros(self.state_shape)
        self.C = torch.zeros(self.state_shape)
        
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

        # Dropouts TODO
        self.H_drop = nn.Dropout2d(p=self.dropout)
        self.C_drop = nn.Dropout2d(p=self.dropout)

        self.apply(initialize_weights)
    
    def forward(self, x, C):
        # TODO
        b_i = 0
        b_f = 0
        b_c = 0
        b_o = 0
        
        i_t = nn.sigmoid(self.W_xi(x) + self.W_hi(self.H) + self.W_ci(self.C) + b_i)
        f_t = nn.sigmoid(self.W_xf(x) + self.W_hf(self.H) + self.W_cf(self.C) + b_f)
        C_t = f_t * self.C + i_t * nn.tanh(self.W_xc(x) + self.W_hc(self.H) + b_c)
        o_t = nn.sigmoid(self.W_xo(x) + self.W_ho(Ht) + self.W_co(self.C) + b_o)
        H_t = o_t * nn.tanh(C_t)
        
        self.C = C_t
        self.H = H_t
        
        return H_t, C_t


class ConvLSTM(nn.Module):

    def __init__(self, input_bands, input_dim, kernels, num_layers, bidirectional, dropout):
        super().__init__()
        self.input_bands = input_bands
        self.input_dim = input_dim
        self.kernels = kernels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        self.layers_fwd = self.initialize_layers()
        self.layers_bwd = None
        if self.bidirectional:
            self.layers_bwd = self.initialize_layers()
        self.fc_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.kernels*self.input_dim**2*(1 if not self.bidirectional else 2), 
                out_features=1024
            ),
            nn.Linear(
                in_features=1024, 
                out_features=1
            )
        )
            
        self.apply(initialize_weights)
        
    def initialize_layers(self):
        """Initialize a single direction of the model's layers.
        
        This function takes care of stacking layers, allocating
        dropout and assigning correct input feature number for
        each layer in the stack."""
        
        layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            layers.append(
                ConvLSTMCell(
                    input_bands=self.input_bands if i == 0 else self.kernels, 
                    input_dim=self.input_dim,
                    dropout=self.dropout if i+1 < self.num_layers else 0,
                    kernels=self.kernels,
                    batch_norm=False
                )
            )
            
        return layers
        
    def forward(self, x):
        """Perform forward pass with the model.
        
        For each item in the sequence, the data is propagated 
        through each layer and both directions, if possible.
        In case of a bidirectional model, the outputs are 
        concatenated from both directions. The output of the 
        last item of the sequence is further given to the FC
        layers to produce the final batch of predictions. 
        
        Args:
            x:  A batch of spatial data sequences. The data
                should be in the following format:
                [Batch, Seq, Band, Dim, Dim]
                    
        Returns:
            A batch of predictions."""
        
        seq_len = x.shape[1]
        
        for seq_idx in range(seq_len):
            
            layer_in_out = x[:,seq_idx,::]
            states = None
            for layer in self.layers_fwd:
                layer_in_out, states = layer(layer_in_out, states)
                
            if not self.bidirectional:
                continue
                
            layer_in_out_bwd = x[:,-seq_idx,::]
            states = None
            for layer in self.layers_bwd:
                layer_in_out_bwd, states = layer(layer_in_out_bwd, states)
            
            layer_in_out = torch.cat((layer_in_out,layer_in_out_bwd),dim=1)
            
        return self.fc_output(layer_in_out)
