from typing import Callable, List, Dict, Optional
import torch
from torch import nn
from .base import AtomicModule
from ..layer import EmbeddingLayer, CutoffLayer
from ..utils import _scatter_add, find_distances


class TwoBody(AtomicModule):

    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 cutoff_fn  : CutoffLayer,
                 k_max: int=12,
                 ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.cutoff_fn = cutoff_fn
        self.k_max = k_max
        self.poly_layer = nn.Linear(embedding_layer.n_channel * 2, k_max)

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        find_distances(batch_data)
        n_atoms = batch_data['atomic_number'].shape[0]
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']
        dij = batch_data['dij']                                      # [n_edges]
        emb = self.embedding_layer(batch_data)     # [n_atoms, n_channel]
        emb_pair = torch.cat((emb[idx_i], emb[idx_j]), dim=1)        # [n_edges, n_channel * 2]
        poly_term = self.poly_layer(emb_pair)                        # [n_edges, k_max]
        poly_features = torch.cat([torch.pow(dij, -i).unsqueeze(-1) for i in range(1, self.k_max + 1)], dim=1)
        energy = torch.sum(poly_term * poly_features, dim=1) * self.cutoff_fn(dij)         # [n_edges]
        batch_data['site_energy'] = _scatter_add(energy, idx_i, dim_size=n_atoms)
        return batch_data
