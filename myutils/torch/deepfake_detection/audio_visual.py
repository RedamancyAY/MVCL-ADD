# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import pytorch_lightning as pl
import torch
import torch.nn as nn


class AudioVisualDeepfakeDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def calcuate_loss(self, batch_res, batch, stage='val'):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def read_data_from_batch(self, batch):
        res = {}
        res["video"] = batch["video"]
        res["audio"] = batch["audio"]
        if "video_label" in batch:
            res["video_label"] = batch["video_label"]
        if "audio_label" in batch:
            res["audio_label"] = batch["audio_label"]

        if "label" in batch:
            res["total_label"] = batch["label"]
        elif "total_label" in batch:
            res["total_label"] = batch["total_label"]
        return res

    def video_post_process(self, video):
        return video

    def audio_post_process(self, audio):
        return audio


    def logit_to_pred(self, x):

        if len(x.shape) == 1 or x.shape[-1] == 1:
            return (torch.sigmoid(x) + 0.5).int()
        elif len(x.shape) == 2 and x.shape[-1] == 2:
            return torch.argmax(x, dim=1)
        else:
            raise ValueError("Error!, wrong logit shape: ", x.shape)
        
    def convert_logits_to_pred(self,batch_res, combine_AVlogits_to_total=False, *kwargs):
        batch_res["video_pred"] = self.logit_to_pred(batch_res["video_logit"])
        batch_res["audio_pred"] = self.logit_to_pred(batch_res["audio_logit"])

        
        if "total_logit" in batch_res:
            batch_res["total_pred"] = self.logit_to_pred(batch_res["total_logit"])
        else:
            batch_res["total_pred"] = torch.bitwise_and(
                batch_res["video_pred"], batch_res["audio_pred"]
            )

        if combine_AVlogits_to_total:
            batch_res["total_pred"] = torch.bitwise_and(
                batch_res["video_pred"], batch_res["audio_pred"]
            )

    def _shared_pred(self, batch, batch_idx, stage="train"):

        import time
        s = time.time()
        
        batch = self.read_data_from_batch(batch)
        video, audio = batch["video"], batch["audio"]

        batch_res = self.model(video=video, audio=audio)

        self.convert_logits_to_pred(batch_res)

        e = time.time()
        # print(e-s, video.shape)
        return batch_res, batch

    def _shared_eval_step(self, batch, batch_idx, stage="train"):
        batch_res, batch = self._shared_pred(batch, batch_idx, stage=stage)

        loss = self.calcuate_loss(batch_res, batch, stage=stage)

        if not isinstance(loss, dict):
            loss = {'loss' : loss}
            
        suffix = ""
        self.log_dict(
            {f"{stage}-{key}{suffix}" : loss[key] for key in loss},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size = batch['total_label'].shape[0]
        )
        batch_res.update(loss)
        return batch_res
        

    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, stage="test")

    def prediction_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, stage="predict")
