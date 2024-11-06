import torch
import torch.distributions as dist
from torch.nn.functional import softplus


#---------------------------------------------------------------
# Logisitic Distribution
#---------------------------------------------------------------
class LogisticDistribution(dist.Distribution):
    def __init__(self, input_size: int):
        super().__init__(validate_args=False)
        self.input_size = input_size
        self.u = dist.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

    def log_prob(self, value):
        # - log(1 + exp(hd)) - log(1 + exp(- hd))
        return - (softplus(value, beta = 1) + softplus(- value, beta = 1))
    
    def sample(self, amount: int, device: torch.DeviceObjType = None):
        device = device or torch.device('cpu') 
        sample = self.u.sample([amount, self.input_size])
        sample = sample.reshape(amount, self.input_size).to(device)
        return torch.log(sample) - torch.log(1. - sample)
    