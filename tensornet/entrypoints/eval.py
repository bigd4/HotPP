from ase.io import read
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensornet.data import ASEData, ASEDBData, atoms_collate_fn
from tqdm import tqdm


def eval(model, data_loader, properties, device):
    output = {prop: [] for prop in properties}
    target = {prop: [] for prop in properties}
    n_atoms = []
    for batch_data in tqdm(data_loader):
        batch_data = {key: value.to(device) for key, value in batch_data.items()}
        model(batch_data, properties, create_graph=False)
        n_atoms.extend(batch_data['n_atoms'].detach().cpu().numpy())
        for prop in properties:
            output[prop].extend(batch_data[f'{prop}_p'].detach().cpu().numpy())
            if f'{prop}_t' in batch_data:
                target[prop].extend(batch_data[f'{prop}_t'].detach().cpu().numpy())
    for prop in properties:
        np.save(f'output_{prop}.npy', np.array(output[prop]))
        np.save(f'target_{prop}.npy', np.array(target[prop]))
    np.save('n_atoms.npy', np.array(n_atoms))
    return None


def main(*args, modelfile='model.pt', indices=None, device='cpu', datafile='data.traj',
         format=None, properties=["energy", "forces"], batchsize=32, num_workers=4, pin_memory=True,
         **kwargs):
    model = torch.load(modelfile, map_location=device)
    model.eval()
    cutoff = float(model.cutoff.detach().cpu().numpy())
    if indices is not None:
        indices = np.loadtxt(indices, dtype=int)

    if '.db' in datafile:
        dataset = ASEDBData(datapath=datafile,
                            indices=indices,
                            properties=properties,
                            cutoff=cutoff)
    else:
        frames = read(datafile, index=':', format=format)
        dataset = ASEData(frames=frames,
                          indices=indices,
                          cutoff=cutoff,
                          properties=properties)

    data_loader = DataLoader(dataset,
                             batch_size=batchsize,
                             shuffle=False,
                             collate_fn=atoms_collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    eval(model, data_loader, properties, device)

if __name__ == "__main__":
    main()
