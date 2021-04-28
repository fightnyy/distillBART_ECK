from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from main import DistillBart
from preprocessing import load_multilingual_dataset
import pytorch_lightning as pl
import os


if __name__ == "__main__":
    # trainer = pl.Trainer(gpus=None)
    trainer = pl.Trainer(
        gpus=-1,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(
                dirpath="./drive/MyDrive/mlbart_ckpt",
                monitor="val_loss",
                filename="paraphrase_mlbart_{epoch:02d}-{val_loss:.2f}",
                save_top_k=-1,
                mode="min",
            ),
        ],
        progress_bar_refresh_rate=20,
    )
    train_dataloader, validation_dataloader = load_multilingual_dataset(
        dataset_path=f"{os.getcwd()}/drive/MyDrive/dataset", batch_size=4
    )
    model = DistillBart(9, 3)
    trainer.fit(
        model, train_dataloader=train_dataloader, val_dataloaders=validation_dataloader
    )
