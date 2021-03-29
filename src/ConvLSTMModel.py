import torch
from torch import nn
from ConvLSTMCell import ConvLSTMCell

class ConvLSTMModel(nn.Module):

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
            x:  A batch of spatial data sequences.   The data
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
