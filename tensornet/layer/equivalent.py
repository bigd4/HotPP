# TODO 
# resnet
# Now segment_coo have some bug in getting second-order derivative
# https://github.com/rusty1s/pytorch_scatter/issues/299
# from torch_scatter import segment_coo

import torch
from torch import nn
from typing import Dict, Callable, Union, Optional
from .base import RadialLayer, CutoffLayer
from .activate import TensorActivateDict
from ..utils import find_distances, find_moment, _scatter_add, _aggregate, expand_to


# input_tensors be like:
#   0: [n_atoms, n_channel]
#   1: [n_atoms, n_channel, n_dim]
#   2: [n_atoms, n_channel, n_dim, n_dim]
#   .....
# coordinate: [n_atoms, n_dim]

__all__ = ["TensorLinear",
           "TensorBiLinear",
           "TensorAggregateLayer",
           "SelfInteractionLayer",
           "NonLinearLayer",
           "SOnEquivalentLayer",
           ]


class TensorLinear(nn.Module):
    """
    Linear layer for tensors with shape [n_batch, n_channel, n_dim, n_dim, ...]
    """
    def __init__(self,
                 input_dim   : int,
                 output_dim  : int,
                 bias        : bool=False,
                 ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, 
                input_tensor: torch.Tensor,   # [n_batch, n_channel, n_dim, n_dim, ...]
                ):
        way = len(input_tensor.shape) - 2
        if way == 0:
            output_tensor = self.linear(input_tensor)
        else:
            input_tensor = torch.transpose(input_tensor, 1, -1)
            output_tensor = self.linear(input_tensor)
            output_tensor = torch.transpose(output_tensor, 1, -1)
        return output_tensor


class TensorBiLinear(nn.Module):
    """
    Linear layer for tensors with shape [n_batch, n_channel, n_dim, n_dim, ...]
    """
    def __init__(self,
                 input_dim   : int,
                 emb_dim     : int,
                 output_dim  : int,
                 bias        : bool=False,
                 ) -> None:
        super().__init__()
        self.linear = nn.Bilinear(input_dim, emb_dim, output_dim, bias=bias)

    def forward(self,
                input_tensor: torch.Tensor,   # [n_batch, n_channel, n_dim, n_dim, ...]
                emb:          torch.Tensor,   # [n_batch, n_emb_channel]
                ):
        way = len(input_tensor.shape) - 2
        if way == 0:
            output_tensor = self.linear(input_tensor, emb)
        else:
            input_tensor = torch.transpose(input_tensor, 1, -1)
            emb = expand_to(emb, way + 2, dim=1).expand([*input_tensor.shape[:-1], -1])
            output_tensor = self.linear(input_tensor, emb)
            output_tensor = torch.transpose(output_tensor, 1, -1)
        return output_tensor


class SimpleTensorAggregateLayer(nn.Module):
    """
    In this type of layer, the rbf mixing only different if r_way different
    """
    def __init__(self,
                 radial_fn      : RadialLayer,
                 n_channel      : int,
                 max_in_way     : int=2,
                 max_out_way    : int=2,
                 max_r_way      : int=2,
                 norm_factor    : float=1.,
                 ) -> None:
        super().__init__()
        # get all possible "i, r, o" combinations
        self.rbf_mixing_dict = nn.ModuleDict()
        self.max_r_way = max_r_way
        self.inout_combinations = {r_way: [] for r_way in range(max_r_way + 1)}
        for r_way in range(max_r_way + 1):
            self.rbf_mixing_dict[str(r_way)] = nn.Linear(radial_fn.n_features, n_channel, bias=False)
            for in_way in range(max_in_way + 1):
                for z_way in range(min(in_way, r_way) + 1):
                    out_way = in_way + r_way - 2 * z_way
                    if out_way <= max_out_way:
                        self.inout_combinations[r_way].append((in_way, out_way))
        self.radial_fn = radial_fn
        self.register_buffer("norm_factor", torch.tensor(norm_factor).float())

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        # These 3 rows are required by torch script
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']

        n_atoms = batch_data['atomic_number'].shape[0]
        _, dij, uij = find_distances(batch_data)
        rbf_ij = self.radial_fn(dij)    # [n_edge, n_rbf]

        for r_way, rbf_mixing in self.rbf_mixing_dict.items():
            r_way = int(r_way)
            fn = rbf_mixing(rbf_ij) # [n_edge, n_channel]
            # TODO: WHY!!!!!!!!!! CAO!
            # fn = fn * input_tensor_dict[0]
            moment_tensor = find_moment(batch_data, r_way)  # [n_edge, n_dim, ...]
            for in_way, out_way in self.inout_combinations[r_way]:
                input_tensor = input_tensors[in_way][idx_j]      # [n_edge, n_channel, n_dim, n_dim, ...]
                output_tensor = _aggregate(moment_tensor, fn, input_tensor, in_way, r_way, out_way)
                output_tensor = _scatter_add(output_tensor, idx_i, dim_size=n_atoms) / self.norm_factor

                if out_way not in output_tensors:
                    output_tensors[out_way] = output_tensor
                else:
                    output_tensors[out_way] += output_tensor
        return output_tensors


class TensorAggregateLayer(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 n_channel      : int,
                 max_in_way     : int=2,
                 max_out_way    : int=2,
                 max_r_way      : int=2,
                 norm_factor    : float=1.,
                 ) -> None:
        super().__init__()
        # get all possible "i, r, o" combinations
        self.all_combinations = {}
        self.rbf_mixing_dict = nn.ModuleDict()
        for in_way in range(max_in_way + 1):
            for r_way in range(max_r_way + 1):
                for z_way in range(min(in_way, r_way) + 1):
                    out_way = in_way + r_way - 2 * z_way
                    if out_way <= max_out_way:
                        comb = (in_way, r_way, out_way)
                        self.all_combinations[str(comb)] = comb
                        self.rbf_mixing_dict[str(comb)] = nn.Linear(radial_fn.n_features, n_channel, bias=False)

        self.radial_fn = radial_fn
        self.register_buffer("norm_factor", torch.tensor(norm_factor).float())

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        # These 3 rows are required by torch script
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']

        n_atoms = batch_data['atomic_number'].shape[0]
        _, dij, uij = find_distances(batch_data)
        rbf_ij = self.radial_fn(dij)    # [n_edge, n_rbf]

        for comb, rbf_mixing in self.rbf_mixing_dict.items():
            in_way, r_way, out_way = self.all_combinations[comb]
            fn = rbf_mixing(rbf_ij) # [n_edge, n_channel]
            # TODO: WHY!!!!!!!!!! CAO!
            # fn = fn * input_tensor_dict[0]
            moment_tensor = find_moment(batch_data, r_way)  # [n_edge, n_dim, ...]
            input_tensor = input_tensors[in_way][idx_j]      # [n_edge, n_channel, n_dim, n_dim, ...]
            output_tensor = _aggregate(moment_tensor, fn, input_tensor, in_way, r_way, out_way)
            output_tensor = _scatter_add(output_tensor, idx_i, dim_size=n_atoms) / self.norm_factor
            if out_way not in output_tensors:
                output_tensors[out_way] = output_tensor
            else:
                output_tensors[out_way] += output_tensor

        return output_tensors


class SelfInteractionLayer(nn.Module):
    def __init__(self,
                 input_dim  : int,
                 max_in_way : int,
                 output_dim : int=10,
                 ) -> None:
        super().__init__()
        # only the way 0 can have bias
        self.linear_interact_list = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        for _ in range(1, max_in_way + 1):
            self.linear_interact_list.append(nn.Linear(input_dim, output_dim, bias=False))

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for way, linear_interact in enumerate(self.linear_interact_list):
            if way in input_tensors:
                # swap channel axis and the last dim axis
                input_tensor = torch.transpose(input_tensors[way], 1, -1)
                output_tensor = linear_interact(input_tensor)
                output_tensors[way] = torch.transpose(output_tensor, 1, -1)
        return output_tensors


# TODO: cat different way together and use Linear layer to got factor of every channel
class NonLinearLayer(nn.Module):
    def __init__(self,
                 max_in_way  : int,
                 input_dim   : int,
                 activate_fn : str='jilu',
                 ) -> None:
        super().__init__()
        self.activate_list = nn.ModuleList([TensorActivateDict[activate_fn](input_dim) for _ in range(max_in_way + 1)])

    def forward(self,
                input_tensors: Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for way, activate in enumerate(self.activate_list):
            if way in input_tensors:
                output_tensors[way] = activate(input_tensors[way])
        return output_tensors


class SOnEquivalentLayer(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='jilu',
                 mode           : str='normal',
                 ) -> None:
        super().__init__()
        if mode == 'normal':
            self.tensor_aggregate = TensorAggregateLayer(radial_fn=radial_fn,
                                                        n_channel=input_dim,
                                                        max_in_way=max_in_way,
                                                        max_out_way=max_out_way,
                                                        max_r_way=max_r_way,
                                                        norm_factor=norm_factor,)
        elif mode == 'simple':
            self.tensor_aggregate = SimpleTensorAggregateLayer(radial_fn=radial_fn,
                                                               n_channel=input_dim,
                                                               max_in_way=max_in_way,
                                                               max_out_way=max_out_way,
                                                               max_r_way=max_r_way,
                                                               norm_factor=norm_factor,)
        # input for SelfInteractionLayer and NonLinearLayer is the output of TensorAggregateLayer
        # so the max_in_way should equal to max_out_way of TensorAggregateLayer
        self.self_interact = SelfInteractionLayer(input_dim=input_dim,
                                                  max_in_way=max_out_way,
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_in_way=max_out_way,
                                         input_dim=output_dim)

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        input_tensors = self.propagate(input_tensors, batch_data)
        return input_tensors

    # TODO: sparse version
    def message_and_aggregate(self,
                              input_tensors : Dict[int, torch.Tensor],
                              batch_data    : Dict[str, torch.Tensor],
                              ) -> Dict[int, torch.Tensor]:
        output_tensors =  self.tensor_aggregate(input_tensors=input_tensors,
                                                batch_data=batch_data)
        # resnet
        for r_way in input_tensors.keys():
            output_tensors[r_way] += input_tensors[r_way]
        return output_tensors

    def update(self,
               input_tensors : Dict[int, torch.Tensor],
               ) -> Dict[int, torch.Tensor]:
        output_tensors = self.self_interact(input_tensors)
        output_tensors = self.non_linear(output_tensors)
        # resnet
        for r_way in input_tensors.keys():
            output_tensors[r_way] += input_tensors[r_way]
        return output_tensors

    def propagate(self,
                  input_tensors : Dict[int, torch.Tensor],
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[int, torch.Tensor]:
        output_tensors = self.message_and_aggregate(input_tensors, batch_data)
        output_tensors = self.update(output_tensors)
        return output_tensors
