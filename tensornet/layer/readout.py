import torch
from torch import nn
from typing import List, Dict, Callable, Any
from .equivalent import TensorLinear, TensorBiLinear
from .activate import TensorActivateDict


__all__ = ["ReadoutLayer"]


class ReadoutMLP(nn.Module):
    def __init__(self,
                 n_dim       : int,
                 way         : int,
                 activate_fn : str="silu",
                 bi          : bool=False,
                 e_dim       : int=0,
                 ) -> None:
        super().__init__()
        self.activate_fn = TensorActivateDict[activate_fn](n_dim)
        self.bi = bi
        if bi:
            self.layer1 = TensorBiLinear(n_dim, e_dim, n_dim, bias=(way==0))
            self.layer2 = TensorBiLinear(n_dim, e_dim, 1, bias=(way==0))
        else:
            self.layer1 = TensorLinear(n_dim, n_dim, bias=(way==0))
            self.layer2 = TensorLinear(n_dim, 1, bias=(way==0))
            
    def forward(self, 
                input_tensor: torch.Tensor,   # [n_batch, n_channel, n_dim, n_dim, ...]
                emb:          torch.Tensor,   # [n_batch, n_channel]
                ):
        if self.bi:
            return self.layer2(self.activate_fn(self.layer1(input_tensor, emb)), emb)
        else:
            return self.layer2(self.activate_fn(self.layer1(input_tensor)))


class ReadoutLayer(nn.Module):

    def __init__(self,
                 n_dim       : int,
                 target_way  : Dict[str, int]={"site_energy": 0},
                 activate_fn : str="silu",
                 bi          : bool=False,
                 e_dim       : int=0,
                 ) -> None:
        super().__init__()
        self.target_way = target_way
        self.layer_dict = nn.ModuleDict({
                prop: nn.Sequential(
                    TensorBiLinear(n_dim, e_dim, n_dim, bias=(way==0)),
                    TensorActivateDict[activate_fn](n_dim),
                    TensorBiLinear(n_dim, e_dim, 1, bias=(way==0)),
                    )
                for prop, way in target_way.items()
                })
        self.layer_dict = nn.ModuleDict({
            prop: ReadoutMLP(n_dim=n_dim, e_dim=e_dim, bi=bi, way=way)
            for prop, way in target_way.items()
            })

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                emb           : Dict[int, torch.Tensor],
                ) -> Dict[str, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[str, torch.Tensor], {})
        for prop, readout_layer in self.layer_dict.items():
            way = self.target_way[prop]
            output_tensors[prop] = readout_layer(input_tensors[way], emb).squeeze(1)  # delete channel dim
        return output_tensors
