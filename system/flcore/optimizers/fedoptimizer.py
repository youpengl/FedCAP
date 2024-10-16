import torch
from torch.optim import Optimizer

class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.0):
        default = dict(lr=lr, lamda=lamda)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['lamda'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])