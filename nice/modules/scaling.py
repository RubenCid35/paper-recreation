import torch

#---------------------------------------------------------------
# Rescaling Layer (section 3.3)
#---------------------------------------------------------------
class ScalingLayer(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        scaling_init = torch.zeros(input_size, requires_grad=True)
        self.S = torch.nn.Parameter(scaling_init)
    
    def forward(self, x, original_det):
        log_det = torch.sum(self.S)
        scaling = torch.diag(torch.exp( self.S ))
        return torch.matmul(x, scaling), original_det + log_det
    
    def inverse(self, y, original_det):
        log_det = torch.sum(self.S)
        scaling = torch.diag(torch.exp( - self.S ))
        return torch.matmul(y, scaling), original_det - log_det
