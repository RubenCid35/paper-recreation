import numpy
import torch
import torch.nn as nn

#---------------------------------------------------------------
# General Coupling Layer
#---------------------------------------------------------------

def create_model(input_size: int, hidden_size: int, n_layers: int = 4, add_norm: bool = False) -> nn.Module:
    layers = []
    sizes = [input_size] + [hidden_size] * n_layers + [input_size]
    for _in, _out in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(_in, _out))
        torch.nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='leaky_relu')
        layers.append(nn.LeakyReLU(0.01))
        if add_norm: layers.append(nn.BatchNorm1d(_out))

    if add_norm:  return nn.Sequential(*layers[:-2]) # remove leaky relu + batch norm
    else       :  return nn.Sequential(*layers[:-1]) # remove leaky relu 

def intercalate(a: torch.Tensor, b: torch.Tensor, mask_order: bool) -> torch.Tensor:
    full_size = a.shape[1] + b.shape[1]
    full_vector = torch.zeros(a.shape[0], full_size, device = a.device)
    if mask_order:
        full_vector[:,  ::2] = a
        full_vector[:, 1::2] = b
    else:
        full_vector[:,  ::2] = b
        full_vector[:, 1::2] = a
    return full_vector

class GeneralCouplingLayer(nn.Module):
    def __init__(self, id: int, input_size: int, hidden_size: int, n_layers: int = 4):
        super().__init__()
        assert input_size % 2 == 0, "The input size must be an even number."

        self.id = id        
        self.mask_order = id % 2 == 0
        self.input_size, self.hidden_size = input_size, hidden_size

        self.add_module('m', create_model(input_size // 2, hidden_size, n_layers, False))

    def forward(self, h: torch.Tensor): 
        # create partition of input
        if self.mask_order: h1, h2 = h[:, ::2], h[:, 1::2]
        else: h2, h1 = h[:, ::2], h[:, 1::2]

        # get representation of each partition
        y1, y2 = h1, self.coupling(h2, self.m(h1))
        return intercalate(y1, y2, self.mask_order) # join partitions together
    
    def inverse(self, h: torch.Tensor): 

        # create partition of input
        if self.mask_order: h1, h2 = h[:, ::2], h[:, 1::2]
        else: h2, h1 = h[:, ::2], h[:, 1::2]

        # get representation of each partition
        y1, y2 = h1, self.inv_coupling(h2, self.m(h1))
        return intercalate(y1, y2, self.mask_order) # join partitions together
    
    def coupling(self, a, b): return NotImplemented("This class is general. Use a specific coupling layer")
    def inv_coupling(self, a, b): return NotImplemented("This class is general. Use a specific coupling layer")

#---------------------------------------------------------------
# Additive Coupling Layer
#---------------------------------------------------------------
class AdditiveCouplingLayer(GeneralCouplingLayer):
    """Additive Coupling Layer.
    ```
        y2 = x2 + m(x1)
        x2 = y2 - m(y1)
    ```
    """
    def coupling(self, a, b): return a + b   
    def inv_coupling(self, a, b): return a - b

#---------------------------------------------------------------
# Multiplicative Coupling Layer
#---------------------------------------------------------------
class MultiplicativeCouplingLayer(GeneralCouplingLayer):
    """Multiplicative Coupling Layer.
    ```
        y2 = x2 * m(x1)
        x2 = y2 / m(y1)
    ```
    """
    def coupling(self, a, b): return torch.mul(a, b)   
    def inv_coupling(self, a, b: torch.Tensor): return torch.mul(a, b.reciprocal())