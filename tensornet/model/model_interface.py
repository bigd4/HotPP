from typing import Optional, Dict, List, Type, Any
from .base import AtomicModule
from ..loss import Loss, MissingValueLoss, ForceScaledLoss
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class LitAtomicModule(pl.LightningModule):

    def __init__(self,
                 model: AtomicModule,
                 p_dict: Dict,
    ):
        super().__init__()
        self.p_dict = p_dict
        self.model = model
        self.loss_calculator = self.get_loss_calculator()

        grad_prop = set(['forces', 'virial', 'stress'])
        self.required_derivatives = len(grad_prop.intersection(self.p_dict["Train"]['targetProp'])) > 0
        # self.save_hyperparameters(ignore=['model'])

    def forward(self,
                batch_data   : Dict[str, torch.Tensor],
                properties   : Optional[List[str]]=None,
                create_graph : bool=True,
                ) -> Dict[str, torch.Tensor]:
        results = self.model(batch_data, properties, create_graph)
        return results

    def get_loss_calculator(self):
        train_dict = self.p_dict['Train']
        target = train_dict['targetProp']
        weight = train_dict['weight']
        weights = {p: w for p, w in zip(target, weight)}
        if "direct_forces" in weights:
            weights["forces"] = weights.pop("direct_forces") # direct forces use the same key of forces
        if train_dict['allowMissing']:
            # TODO: rewrite loss function
            if train_dict['forceScale'] > 0:
                raise Exception("Now forceScale not support allowMissing!")
            return MissingValueLoss(weights, loss_fn=F.mse_loss)
        else:
            if train_dict['forceScale'] > 0:
                return ForceScaledLoss(weights, loss_fn=F.mse_loss, scaled=train_dict['forceScale'])
            else:
                return Loss(weights, loss_fn=F.mse_loss)

    def training_step(self, batch, batch_idx):
        self.model(batch, self.p_dict["Train"]['targetProp'])
        loss, loss_dict = self.loss_calculator.get_loss(batch, verbose=True)
        self.log("train_loss", loss)
        for prop in loss_dict:
            self.log(f'train_{prop}', loss_dict[prop])
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.required_derivatives)
        self.model(batch, self.p_dict["Train"]['targetProp'], create_graph=False)
        loss, loss_dict = self.loss_calculator.get_loss(batch, verbose=True)
        self.log("val_loss", loss, batch_size=batch['n_atoms'].shape[0])
        for prop in loss_dict:
            self.log(f'val_{prop}', loss_dict[prop], batch_size=batch['n_atoms'].shape[0])

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.required_derivatives)
        self.model(batch, self.p_dict["Train"]['targetProp'], create_graph=False)
        loss, loss_dict = self.loss_calculator.get_loss(batch, verbose=True)
        loss_dict['test_loss'] = loss
        self.log_dict(loss_dict)
        return loss_dict

    def get_optimizer(self):
        opt_dict = self.p_dict["Train"]["Optimizer"]
        decay_interactions = {}
        no_decay_interactions = {}
        embedding = {}
        readout = {}
        for name, param in self.model.named_parameters():
            if "son_equivalent_layers" in name:
                if "weight" in name:
                    decay_interactions[name] = param
                else:
                    no_decay_interactions[name] = param
            else:
                if "embedding" in name:
                    embedding[name] = param
                if "readout" in name:
                    readout[name] = param

        param_options = dict(
            params=[
                {
                    "name": "embedding",
                    "params": list(embedding.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "interactions_decay",
                    "params": list(decay_interactions.values()),
                    "weight_decay": opt_dict["weightDecay"],
                },
                {
                    "name": "interactions_no_decay",
                    "params": list(no_decay_interactions.values()),
                    "weight_decay": 0.0,
                },
                {
                    "name": "readouts",
                    "params": list(readout.values()),
                    "weight_decay": 0.0,
                },
            ],
            lr=self.p_dict["Train"]["learningRate"],
            amsgrad=opt_dict["amsGrad"],
        )

        if opt_dict['type'] == "Adam":
            return torch.optim.Adam(**param_options)
        elif opt_dict['type'] == "AdamW":
            return torch.optim.AdamW(**param_options)
        else:
            raise Exception("Unsupported optimizer: {}!".format(opt_dict["type"]))

    def get_lr_scheduler(self, optimizer):
        lr_dict = self.p_dict["Train"]["LrScheduler"]
        if lr_dict['type'] == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=lr_dict['gamma']
                )
        elif lr_dict['type'] == "reduceOnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    factor=lr_dict['lrFactor'],
                    patience=lr_dict['patience']
                    )
            if self.p_dict["Train"]["evalEpochInterval"] == 1:
                lr_scheduler_config = {
                        "scheduler": scheduler,
                        "interval": "step",
                        "monitor": "val_loss",
                        "frequency": self.p_dict["Train"]["evalStepInterval"],
                        }
            else:
                lr_scheduler_config = {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "monitor": "val_loss",
                        "frequency": self.p_dict["Train"]["evalEpochInterval"],
                        }
            return lr_scheduler_config
        elif lr_dict['type'] == "constant":
            return None
        else:
            raise Exception("Unsupported LrScheduler: {}!".format(lr_dict['type']))

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_lr_scheduler(optimizer)
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # Learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.p_dict["Train"]["warmupSteps"]:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.p_dict["Train"]["warmupSteps"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.p_dict["Train"]["learningRate"]

    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value
