
import pytorch_lightning as pl
import torch
import torch.nn as nn


class BinaryClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def configure_loss_fn(self):
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=None)

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            loss = self.loss_fn(batch_res["logit"], label.long().flatten())
        elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    
    def configure_normalizer(self):
        return None
    
    def normalize_input(self, x):
        if not hasattr(self, "normalizer"):
            self.normalizer = self.configure_normalizer()
        
        if self.normalizer is not None:
            x = self.normalizer(x)
        return x
            
    def configure_optimizers(self):
        raise NotImplementedError

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        raise NotImplementedError

    def _shared_eval_step(
        self, batch, batch_idx, stage="train", dataloader_idx=0, *args, **kwargs
    ):

        try:
            batch_res = self._shared_pred(batch, batch_idx, stage=stage)
        except TypeError:
            batch_res = self._shared_pred(batch, batch_idx)
            
        
        label = batch["label"]
        loss = self.calcuate_loss(batch_res, batch)

        if not isinstance(loss, dict):
            loss = {'loss' : loss}
            
        suffix = "" if dataloader_idx == 0 else f"-dl{dataloader_idx}"
        self.log_dict(
            {f"{stage}-{key}{suffix}" : loss[key] for key in loss},
            # on_step=True if stage=='train' else False,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size = batch['label'].shape[0]
        )
        batch_res.update(loss)
        return batch_res

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(
            batch, batch_idx, stage="val", dataloader_idx=dataloader_idx
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(
            batch, batch_idx, stage="test", dataloader_idx=dataloader_idx
        )

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(
            batch, batch_idx, stage="predict", dataloader_idx=dataloader_idx
        )




class DeepfakeAudioClassification(BinaryClassification):
    
    
    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # print(audio.shape)
        
        # out = self.model(audio)

        feature = self.model.extract_feature(audio)
        out = self.model.make_prediction(feature)
        
        out = out.squeeze(-1)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        return {
            "logit": out,
            "pred": batch_pred,
            "feature": feature
        }
