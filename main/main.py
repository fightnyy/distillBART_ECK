from make_config import start
from typing import Dict, Tuple, List
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from bart_tokenizer import AsianBartTokenizer
from asian_bart import AsianBartForConditionalGeneration

import pytorch_lightning as pl
import torch


class DistillBart(pl.LightningModule):

    def __init__(self, num_encoder: int, num_decoder: int):
        super().__init__()
        self.lr = 3e-5
        self.weight_decay = 1e-4,
        self.tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
        self.model = start(num_encoder, num_decoder)
        print("well_loaded")

    def forward(self, batch):
        s1, s2, lang_code = batch
        model_inputs = self.tokenizer.prepare_seq2seq_batch(
            src_texts=s1,
            src_langs=lang_code,
            tgt_texts=s2,
            tgt_langs=lang_code,
            padding="max_length",
            max_len=256,
        )
        for key, v in model_inputs.items():
            model_inputs[key] = model_inputs[key].to("cuda")

        out = self.model(input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'],
                         labels=model_inputs['labels'])
        return out

    def training_step(self, batch, batch_idx):
        """
        Training_Step

        batch : Data from DataLoader
        batch_idx : idx of Data
        """
        out = self.forward(batch)
        loss = out["loss"]
        self.log("train_loss", loss)
        return loss

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
        """configure optimizers and lr schedulers"""
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        return {"optimizer": optimizer}

