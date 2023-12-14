import torch
from torch import nn
import numpy as np
from .base import CutoffLayer

__all__ = ["CosineCutoff",
           "SmoothCosineCutoff",
           "PolynomialCutoff",
           ]


class CosineCutoff(CutoffLayer):
    def forward(self,
                distances : torch.Tensor,
                ) -> torch.Tensor:
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class SmoothCosineCutoff(CutoffLayer):
    def __init__(self,
                 cutoff        : float,
                 cutoff_smooth : float,
                 ) -> None:
        assert cutoff_smooth < cutoff, "'cutoff_smooth must smaller than cutoff"
        super(SmoothCosineCutoff, self).__init__(cutoff)
        self.register_buffer("cutoff_smooth", torch.tensor(cutoff_smooth).float())

    def forward(self, 
                distances : torch.Tensor,
                ) -> torch.Tensor:
        phase = (distances.clamp(min=self.cutoff_smooth, max=self.cutoff) - self.cutoff_smooth)/\
                 (self.cutoff - self.cutoff_smooth) * np.pi
        cutoffs = 0.5 * (torch.cos(phase) + 1.0)
        return cutoffs


# class PolynomialCutoff(CutoffLayer):
#     def forward(self, 
#                 distances : torch.Tensor,
#                 ) -> torch.Tensor:
#         cutoffs = (1.0 - (distances / self.cutoff) ** 2) ** 3
#         cutoffs *= (distances < self.cutoff).float()
#         return cutoffs

class PolynomialCutoff(CutoffLayer):
    def __init__(self,
                 cutoff   : float,
                 p        : int,
                 ) -> None:
        super(PolynomialCutoff, self).__init__(cutoff)
        self.register_buffer("p", torch.tensor(p).float())

    def forward(self,
                distances : torch.Tensor,
                ) -> torch.Tensor:
        x = distances / self.cutoff
        cutoffs = 1.0 - (self.p + 1) * (self.p + 2) / 2 * torch.pow(x, self.p)\
                      + self.p * (self.p + 2) * torch.pow(x, self.p + 1)\
                      - self.p * (self.p + 1) / 2 * torch.pow(x, self.p + 2)
        cutoffs *= (x < 1).float()
        return cutoffs
