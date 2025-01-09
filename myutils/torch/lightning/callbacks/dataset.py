# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
import pytorch_lightning as pl


# + tags=[]
class ChangeImageSizeAfterEpochs(pl.callbacks.Callback):
    def __init__(self, min_epochs, datasets, new_img_size, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs
        self.datasets = datasets
        self.new_img_size = new_img_size

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch == self.min_epochs:
            for ds in self.datasets:
                ds.set_img_size(self.new_img_size)

