import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, out_dim,p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.out_dim = out_dim

    def forward(self, x):
        """
        x_input: B, N, D_local
        x_output: B, D_global
        """
        dim = int(x.size()[2] / self.out_dim)
        # This implicitly applies ReLU on x (clamps negative values)
        x_out = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), dim)).pow(1./self.p)
        # x_out = F.avg_pool2d(x.pow(self.p), (x.size(-2), dim)).pow(1./self.p)
        x_out = x_out.squeeze(-2)
        return x_out