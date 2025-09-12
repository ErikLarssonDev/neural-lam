import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from neural_lam.models.ar_model import ARModel
from neural_lam import constants
from neural_lam.models.wavelet_convolution import WaveConv2dCwt, WaveConv2d

class WNO2d(ARModel):
    """
    A new auto-regressive weather forecasting model
    """
    def __init__(self, args, width=20, level=2, layers=4, size=[268, 238], wavelet=['near_sym_b', 'qshift_b'], in_channel=50, xgrid_range=[0, 238], ygrid_range=[0, 268], padding=0):
        super().__init__(args)

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: list of strings for 2D, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        # sub = 2**4 # 2**4 for 4^o, # 2**3 for 2^o
        # h = int(((721 - 1)/sub))
        # s = int(((1441 - 1)/sub))

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size # [h, s]
        self.wavelet1 = wavelet[0]
        self.wavelet2 = wavelet[1]
        self.in_channel = in_channel
        self.xgrid_range = xgrid_range
        self.ygrid_range = ygrid_range
        self.padding = padding

        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2dCwt(self.width, self.width, self.level, self.size,
                                            self.wavelet1, self.wavelet2) )
            print(f"Conv layer {i} shape: {self.width}")
            self.w.append( nn.Conv2d(self.width, self.width, 1, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        # Some dimensionalities that can be useful to have stored
        self.input_dim = 2*constants.grid_state_dim + constants.grid_forcing_dim +\
            constants.batch_static_feature_dim
        self.output_dim = constants.grid_state_dim

    def process(self, x):
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        # grid = self.get_grid(x.shape, x.device)
        # print(f"Grid shape: {grid.shape}")
        # print(f"Input shape: {x.shape}")
        # x = torch.cat((x, grid), dim=-1)    
        # print(f"Input shape after grid concat: {x.shape}")
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            # print(f"x shape: {x.shape}")
            # print(f"Conv layer {index}")
            # print(f"convl shape: {convl(x).shape}")
            # print("wl shape: ", wl(x).shape)

            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.gelu( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(self.xgrid_range[0], self.xgrid_range[1], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(self.ygrid_range[0], self.ygrid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device) 

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Predict weather state one time step ahead
        X_{t-1}, X_t -> X_t+1

        prev_state: (B, N_grid, d_state), weather state X_t at time t
        prev_prev_state: (B, N_grid, d_state), weather state X_{t-1} at time t-1
        batch_static_features: (B, N_grid, batch_static_feature_dim), static forcing
        forcing: (B, N_grid, forcing_dim), dynamic forcing

        Returns:
        next_state: (B, N_grid, d_state), predicted weather state X_{t+1} at time t+1
        pred_std: None or (B, N_grid, d_state), predicted standard-deviations
                    (pred_std can be ignored by just returning None)
        """

        # Reshape 1d grid to 2d image
        input_flat = torch.cat((prev_state, prev_prev_state, forcing), dim=-1) # (B, N_grid, d_input)
        input_grid = torch.reshape(input_flat, (-1, *constants.grid_shape,
            input_flat.shape[2])) # (B, N_x, N_y, d_input)
        # Most computer vision methods in torch want channel dimension first
        input_grid = input_grid.permute((0,3,1,2)).contiguous() # (B, d_input, N_x, N_y)

        # TODO: Feed input_grid through some model to predict output_grid
        # output_grid = self.layer(input_grid) # Shape (B, d_state, N_x, N_y)
        # print(f"Input grid shape: {input_grid.shape}")
        output_grid = self.process(input_grid) # Shape: Batch * x * y * Channel

        # Reshape back from 2d to flattened grid dimension
        # output_grid = output_grid.permute((0,2,3,1)) # (B, N_x, N_y, d_state)
        next_state = output_grid.flatten(1,2) # (B, N_grid, d_state)

        return next_state, None

if __name__ == "__main__":
    input_dim = 2*constants.grid_state_dim + constants.grid_forcing_dim +\
        constants.batch_static_feature_dim
    
    print(f"Input dimension: {input_dim}")

    output_dim = constants.grid_state_dim

    print(f"Output dimension: {output_dim}")