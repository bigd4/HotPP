import torch
from torch import nn
from typing import List, Dict
from tensornet.utils import _scatter_add


class AtomicModule(nn.Module):
    def forward(self, 
                batch_data   : Dict[str, torch.Tensor],
                properties   : List[str]=['energy'],
                create_graph : bool=True,
                ) -> Dict[str, torch.Tensor]:
        if 'forces' in properties:
            batch_data['coordinate'].requires_grad_()
        site_energy = self.get_site_energy(batch_data)
        if 'site_energy' in properties:
            batch_data['site_energy_p'] = site_energy
        if 'energy' in properties:
            batch_data['energy_p'] = _scatter_add(site_energy, batch_data['batch'])
        if 'forces' in properties:
            batch_data['forces_p'] = -torch.autograd.grad(site_energy.sum(),
                                                          batch_data['coordinate'],
                                                          create_graph=create_graph)[0]
        return batch_data

    def get_site_energy(self):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'get_site_energy'!")