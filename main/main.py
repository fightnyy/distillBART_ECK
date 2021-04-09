from .make_config import start
from typing import Dict,Tuple,List
from torch.optim import Optimizer, AdamW


import pytorch_lightning as pl
import torch


class DistillBart(pl.LightningModule):

    def __init__(self, n_encoder:int, n_decoder:int):
        super().__init()
        self.batch_size = 4
        self.lr =3e-5
        self.model = start(n_encoder, n_decoder)

    def forward(self, batch):
        model_inputs, labels = batch
        out = self.model(**model_inputs, labels = labels)
        return out

    def training_step(self, batch, batch_idx):
        """
        Training_Step

        batch : Data from DataLoader
        batch_idx : idx of Data
        """
        out = self.forward(batch)
        loss = out["loss"]
        self.log("train_loss",loss)
        return out

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> Dict:
        """
        Validation steps

        batch : ([s1, s2, lang_code, lang_code]) Data from DataLoader
        batch_idx : idx of Data
        """
        out = self.forward(batch)
        loss = out["loss"]
        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss



    def configure_optimizers(self):
        optimizer = AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        return {"optimizer": optimizer}

