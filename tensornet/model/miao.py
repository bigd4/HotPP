from typing import Callable, List, Dict, Optional
import torch
from torch import nn
from .base import AtomicModule
from ..layer import SOnEquivalentLayer, EmbeddingLayer, RadialLayer, ReadoutLayer, CutoffLayer
from ..utils import expand_para, find_distances


class MiaoNet(AtomicModule):
    """
    Miao nei ga
    duo xi da miao nei
    """
    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 n_layers        : int,
                 max_r_way       : int or List,
                 max_out_way     : int or List,
                 output_dim      : int or List,
                 activate_fn     : str="jilu",
                 target_way      : Dict[str, int]={"site_energy": 0},
                 mean            : float=0.,
                 std             : float=1.,
                 norm_factor     : float=1.,
                 mode            : str='normal',
                 bi              : bool=True,
                 ):
        super().__init__(mean=mean, std=std)
        self.register_buffer("norm_factor", torch.tensor(norm_factor).float())
        self.embedding_layer = embedding_layer
        max_r_way = expand_para(max_r_way, n_layers)
        max_out_way = expand_para(max_out_way, n_layers)
        max_in_way = [0] + max_out_way[1:]
        hidden_nodes = [embedding_layer.n_channel] + expand_para(output_dim, n_layers)
        self.son_equivalent_layers = nn.ModuleList([
            SOnEquivalentLayer(activate_fn=activate_fn,
                               radial_fn=radial_fn.replicate(),
                               # Use factory method, so the radial_fn in each layer are different
                               max_r_way=max_r_way[i],
                               max_in_way=max_in_way[i],
                               max_out_way=max_out_way[i],
                               input_dim=hidden_nodes[i],
                               output_dim=hidden_nodes[i + 1],
                               norm_factor=norm_factor,
                               mode=mode) for i in range(n_layers)])
        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                          target_way=target_way,
                                          activate_fn=activate_fn,
                                          bi=bi,
                                          e_dim=embedding_layer.n_channel)

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        find_distances(batch_data)
        emb = self.embedding_layer(batch_data=batch_data)
        output_tensors = {0: emb}
        for son_equivalent_layer in self.son_equivalent_layers:
            output_tensors = son_equivalent_layer(output_tensors, batch_data)
        output_tensors = self.readout_layer(output_tensors, emb)
        return output_tensors
