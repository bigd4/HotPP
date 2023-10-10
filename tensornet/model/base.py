import torch
from torch import nn
from typing import List, Dict, Optional
from tensornet.utils import _scatter_add, find_distances, add_scaling


class AtomicModule(nn.Module):

    def __init__(self,
                 mean  : float=0.,
                 std   : float=1.,
                 ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float())
        self.register_buffer("std", torch.tensor(std).float())

    def forward(self,
                batch_data   : Dict[str, torch.Tensor],
                properties   : Optional[List[str]]=None,
                create_graph : bool=True,
                ) -> Dict[str, torch.Tensor]:
        # Use properties=None instead of properties=['energy'] because
        # Mutable default parameters are not supported because
        # Python binds them to the function and they persist across function calls.
        if properties is None:
            properties = ['energy']
        required_derivatives = torch.jit.annotate(List[str], [])
        if 'forces' in properties:
            required_derivatives.append('coordinate')
            batch_data['coordinate'].requires_grad_()
        if 'virial' in properties or 'stress' in properties:
            required_derivatives.append('scaling')
            batch_data['scaling'].requires_grad_()
            add_scaling(batch_data)
        output_tensors = self.calculate(batch_data)
        if 'site_energy' in output_tensors:
            site_energy = output_tensors['site_energy'] * self.std + self.mean
        #######################################
        # for torch.jit.script 
        else:
            site_energy = batch_data['n_atoms']
        #######################################
        if 'dipole' in output_tensors:
            batch_data['dipole_p'] = _scatter_add(output_tensors['dipole'], batch_data['batch'])
        if 'polar_00' in output_tensors:
            polar_00 = _scatter_add(output_tensors['polar_00'], batch_data['batch'])
            polar_11 = _scatter_add(output_tensors['polar_11'], batch_data['batch'])
            polar_22 = _scatter_add(output_tensors['polar_22'], batch_data['batch'])
            polar_off_diagonal = _scatter_add(output_tensors['polar_off_diagonal'], batch_data['batch'])
            polar = polar_off_diagonal + polar_off_diagonal.transpose(1, 2)
            polar[:, 0, 0] += polar_00
            polar[:, 1, 1] += polar_11
            polar[:, 2, 2] += polar_22
            batch_data['polarizability_p'] = polar 

        if 'direct_forces' in output_tensors:
            batch_data['forces_p'] = output_tensors['direct_forces'] * self.std
        if ('site_energy' in properties) or ('energies' in properties):
            batch_data['site_energy_p'] = site_energy
        if 'energy' in properties:
            batch_data['energy_p'] = _scatter_add(site_energy, batch_data['batch'])
        if len(required_derivatives) > 0:
            grads = torch.autograd.grad([site_energy.sum()],
                                        [batch_data[prop] for prop in required_derivatives],
                                        create_graph=create_graph)
        #######################################
        # for torch.jit.script 
        else:
            grads = torch.jit.annotate(List[Optional[torch.Tensor]], [])
        #######################################
        if 'forces' in properties:
            #######################################
            # for torch.jit.script 
            dE_dr = grads[required_derivatives.index('coordinate')]
            if dE_dr is not None:
                batch_data['forces_p'] = -dE_dr
            #######################################
        if 'virial' in properties or 'stress' in properties:
            #######################################
            # for torch.jit.script 
            dE_dl = grads[required_derivatives.index('scaling')]
            if dE_dl is not None:
                batch_data['virial_p'] = -dE_dl
            #######################################
        return batch_data

    def calculate(self):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'calculate'!")

