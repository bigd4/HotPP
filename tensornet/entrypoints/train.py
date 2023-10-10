import logging, time, yaml, os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from ase.data import atomic_numbers
from ..utils import setup_seed
from ..model import MiaoNet, LitAtomicModule
from ..layer.cutoff import *
from ..layer.embedding import AtomicEmbedding
from ..layer.radial import *
from ..data import LitAtomsDataset


torch.set_float32_matmul_precision("high")
log = logging.getLogger(__name__)


class SaveModelCheckpoint(ModelCheckpoint):
    """
    Saves model.pt for eval
    """
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        modelpath = filepath[:-4] + "pt"
        if trainer.is_global_zero:
            torch.save(trainer.lightning_module.model, modelpath)

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        modelpath = filepath[:-4] + "pt"
        if trainer.is_global_zero:
            if os.path.exists(modelpath):
                os.remove(modelpath)


def update_dict(d1, d2):
    for key in d2:
        if key in d1 and isinstance(d1[key], dict):
            update_dict(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


def get_stats(data_dict, dataset):

    if type(data_dict["mean"]) is float:
        mean = data_dict["mean"]
    else:
        try:
            mean = dataset.per_energy_mean.detach().cpu().numpy()
        except:
            mean = 0.

    if data_dict["std"] == "force":
        std = dataset.forces_std.detach().cpu().numpy()
    elif data_dict["std"] == "energy":
        std = dataset.per_energy_std.detach().cpu().numpy()
    else:
        assert type(data_dict["std"]) is float, "std must be 'force', 'energy' or a float!" 
        std = data_dict["std"]

    if type(data_dict["nNeighbor"]) is float:
        n_neighbor = data_dict["nNeighbor"]
    else:
        n_neighbor = dataset.n_neighbor_mean.detach().cpu().numpy()

    if isinstance(data_dict["elements"], list):
        elements = data_dict["elements"]
    else:
        elements = list(dataset.all_elements.detach().cpu().numpy())

    log.info(f"mean  : {mean}")
    log.info(f"std   : {std}")
    log.info(f"n_neighbor   : {n_neighbor}")
    log.info(f"all_elements : {elements}")
    return mean, std, n_neighbor, elements


def get_cutoff(p_dict):
    cutoff = p_dict['cutoff']
    cut_dict = p_dict['Model']['CutoffLayer']
    if cut_dict['type'] == "cos":
        return CosineCutoff(cutoff=cutoff)
    elif cut_dict['type'] == "cos2":
        return SmoothCosineCutoff(cutoff=cutoff, cutoff_smooth=cut_dict['smoothCutoff'])
    elif cut_dict['type'] == "poly":
        return PolynomialCutoff(cutoff=cutoff, p=cut_dict['p'])
    else:
        raise Exception("Unsupported cutoff type: {}, please choose from cos, cos2, and poly!".format(cut_dict['type']))
    

def get_radial(p_dict, cutoff_fn):
    cutoff = p_dict['cutoff']
    radial_dict = p_dict['Model']['RadialLayer']
    if "bessel" in radial_dict['type']:
        radial_fn = BesselPoly(r_max=cutoff, n_max=radial_dict['nBasis'], cutoff_fn=cutoff_fn)
    elif "chebyshev" in radial_dict['type']:
        if "minDist" in radial_dict:
            r_min = radial_dict['minDist']
        else:
            r_min = 0.5
            log.warning("You are using chebyshev poly as basis function, but does not given 'minDist', "
                        "this may cause some problems!")
        radial_fn = ChebyshevPoly(r_max=cutoff, r_min=r_min, n_max=radial_dict['nBasis'], cutoff_fn=cutoff_fn)
    else:
        raise Exception("Unsupported radial type: {}!".format(radial_dict['type']))
    if "MLP" in radial_dict['type']:
        if radial_dict["activateFn"] == "silu":
            activate_fn = nn.SiLU()
        elif radial_dict["activateFn"] == "relu":
            activate_fn = nn.ReLU()
        else:
            raise Exception("Unsupported activate function in radial type: {}!".format(radial_dict["activateFn"]))
        return MLPPoly(n_hidden=radial_dict['nHidden'], radial_fn=radial_fn, activate_fn=activate_fn)
    else:
        return radial_fn


def get_model(p_dict, elements, mean, std, n_neighbor):
    model_dict = p_dict['Model']
    target = p_dict['Train']['targetProp']
    target_way = {}
    if ("energy" in target) or ("forces" in target) or ("virial" in target):
        target_way["site_energy"] = 0
    if "dipole" in target:
        target_way["dipole"] = 1
    if "polarizability" in target:
        target_way["polar_00"] = 0
        target_way["polar_11"] = 0
        target_way["polar_22"] = 0
        target_way["polar_off_diagonal"] = 2
    if "direct_forces" in target:
        assert "forces" not in target_way, "Cannot learn forces and direct_forces at the same time"
        target_way["direct_forces"] = 1
    cut_fn = get_cutoff(p_dict)
    emb = AtomicEmbedding(elements, model_dict['nEmbedding'])  # only support atomic embedding now
    radial_fn = get_radial(p_dict, cut_fn)
    model = MiaoNet(embedding_layer=emb,
                    radial_fn=radial_fn,
                    n_layers=model_dict['nLayer'],
                    max_r_way=model_dict['maxRWay'],
                    max_out_way=model_dict['maxOutWay'],
                    output_dim=model_dict['nHidden'],
                    activate_fn=model_dict['activateFn'],
                    target_way=target_way,
                    mean=mean,
                    std=std,
                    norm_factor=n_neighbor,
                    mode=model_dict['mode']).to(p_dict['device'])
    return model


def main(*args, input_file='input.yaml', load_model=None, load_checkpoint=None, **kwargs):
    # Default values
    p_dict = {
        "workDir": os.getcwd(),
        "seed": np.random.randint(0, 100000000),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "outputDir": os.path.join(os.getcwd(), "outDir"),
        "Data": {
            "path": os.getcwd(),
            "trainBatch": 32,
            "testBatch": 32,
            "std": "force",
            "mean": None,
            "nNeighbor": None,
            "elements": None,
            "numWorkers": 0,
            "pinMemory": False,
        },
        "Model": {
            "mode": "normal",
            "activateFn": "silu",
            "nEmbedding": 64,
            "nLayer": 5,
            "maxRWay": 2,
            "maxOutWay": 2,
            "nHidden": 64,
            "targetWay": {0 : 'site_energy'},
            "CutoffLayer": {
                "type": "poly",
                "p": 5,
            },
            "RadialLayer": {
                "type": "besselMLP",
                "nBasis": 8,
                "nHidden": [64, 64, 64],
                "activateFn": "silu",
            }
        },
        "Train": {
            "maxEpoch": 10000,
            "maxStep": 1000000,
            "learningRate": 0.001,
            "allowMissing": False,
            "targetProp": ["energy", "forces"],
            "weight": [0.1, 1.0],
            "forceScale": 0.,
            "evalStepInterval": 50,
            "evalEpochInterval": 1,
            "logInterval": 50,
            "saveStart": 1000,
            "evalTest": True,
            "gradClip": None,
            "Optimizer": {
                "type": "Adam",
                "amsGrad": True,
                "weightDecay": 0.,
                },
            "LrScheduler": {
                "type": "constant",
            },
            "emaDecay": 0.,
        },
    }
    with open(input_file) as f:
        update_dict(p_dict, yaml.load(f, Loader=yaml.FullLoader))

    os.makedirs(p_dict["outputDir"], exist_ok=True)

    with open("allpara.yaml", "w") as f:
        yaml.dump(p_dict, f)

    setup_seed(p_dict["seed"])
    log.info("Using seed {}".format(p_dict["seed"]))

    log.info(f"Preparing data...")
    dataset = LitAtomsDataset(p_dict)
    dataset.setup()
    mean, std, n_neighbor, elements = get_stats(p_dict["Data"], dataset)

    if load_model is not None and 'ckpt' not in load_model:
        log.info(f"Load model from {load_model}")
        model = torch.load(load_model)
    else:
        model = get_model(p_dict, elements, mean, std, n_neighbor)
        model.register_buffer('all_elements', torch.tensor(elements, dtype=torch.long))
        model.register_buffer('cutoff', torch.tensor(p_dict["cutoff"], dtype=torch.float64))

    if load_model is not None and 'ckpt' in load_model:
        lit_model = LitAtomicModule.load_from_checkpoint(load_model, model=model, p_dict=p_dict)
    else:
        lit_model = LitAtomicModule(model=model, p_dict=p_dict)

    logger = pl.loggers.TensorBoardLogger(save_dir=p_dict["outputDir"])
    callbacks = [
        SaveModelCheckpoint(
            dirpath=p_dict["outputDir"],
            filename='{epoch}-{step}-{val_loss:.4f}',
            save_top_k=5,
            monitor="val_loss"
        ),
        LearningRateMonitor()
    ]
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir='.',
        max_epochs=p_dict["Train"]["maxEpoch"],
        max_steps=p_dict["Train"]["maxStep"],
        enable_progress_bar=False,
        log_every_n_steps=p_dict["Train"]["logInterval"],
        val_check_interval=p_dict["Train"]["evalStepInterval"],
        check_val_every_n_epoch=p_dict["Train"]["evalEpochInterval"],
        gradient_clip_val=p_dict["Train"]["gradClip"],
        )
    if load_checkpoint is not None:
        log.info(f"Load checkpoints from {load_checkpoint}")
        trainer.fit(lit_model, datamodule=dataset, ckpt_path=load_checkpoint)
    else:
        trainer.fit(lit_model, datamodule=dataset)

if __name__ == "__main__":
    main()
