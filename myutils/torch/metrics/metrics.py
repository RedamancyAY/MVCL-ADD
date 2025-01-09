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

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
)


class Classification_Metrics(object):
    def __init__(self, metrics, num_classes=2):
        super().__init__()
        metrics = [_m.lower() for _m in metrics]
        self.metrics = metrics
        self.num_classes = num_classes
        self.metric_classes = []
        for _m in metrics:
            assert _m in ["acc", "auc", "f1"]
            self.metric_classes.append(self._get_metric_from_name(_m))

        # print(self.metric_classes)
            
    def _get_metric_from_name(self, metric_name, cuda=False):
        if metric_name == "acc":
            if self.num_classes > 2:
                res = MulticlassAccuracy(num_classes=self.num_classes, top_k=1)
            else:
                res = BinaryAccuracy()
            
        if metric_name == "auc":
            if self.num_classes > 2:
                res = MulticlassAUROC(num_classes=self.num_classes, top_k=1)
            else:
                res = BinaryAUROC()
 
        return res
                
    
    def _check_device(self, preds):
        if preds.device != self.metric_classes[0].device:
            for i in range(len(self.metrics)):
                self.metric_classes[i].to(preds.device)
    
    def deal_preds(self, preds, target):
        if preds.shape != target.shape:
            preds = torch.argmax(preds, dim=-1)
        return preds

    def __call__(self, preds, target, prefix=""):
        self._check_device(preds)
        preds = self.deal_preds(preds, target)
        res = {}
        for _m, _cls in zip(self.metrics, self.metric_classes):
            res[prefix + _m] = _cls(preds, target)
        return res

    def update(self, preds, target):
        self._check_device(preds)
        preds = self.deal_preds(preds, target)
        for _m, _cls in zip(self.metrics, self.metric_classes):
            _cls.update(preds, target)

    def reset(self):
        for _cls in self.metric_classes:
            _cls.reset()

    def compute(self, prefix=""):
        res = {}
        for _m, _cls in zip(self.metrics, self.metric_classes):
            res[prefix + _m] = _cls.compute()
        return res


# + tags=["active-ipynb", "style-student"]
# my_metrics = Classification_Metrics(["acc", "auc"], num_classes=2)
# my_metrics.reset()
# target = torch.tensor([0, 0, 1, 1])
# preds = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]])
# print(my_metrics(preds, target))
#
# target2 = torch.tensor([0, 1, 1])
# preds2 = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]])
# my_metrics.update(preds2, target2)
#
# my_metrics.compute(prefix="train_")
# -

class MultiTheme_Classification_Metrics(object):
    def __init__(self, metrics, themes, num_classes=2):
        super().__init__()
        self.themes = themes
        self.metrics = metrics
        self.classification_metrics = [
            Classification_Metrics(metrics=metrics, num_classes=num_classes)
            for _ in themes
        ]

    def __call__(self, preds, target):
        assert len(preds) == len(target) and len(preds) == len(self.themes)
        res = {}
        for i in range(len(preds)):
            _res = self.classification_metrics[i](
                preds[i], target[i], prefix=self.themes[i] + "_"
            )
            res.update(_res)
        return res

    def update(self, preds, targets):
        assert len(preds) == len(targets) and len(preds) == len(self.themes)
        for i in range(len(preds)):
            self.classification_metrics[i].update(preds[i], targets[i])

    def update_theme(self, theme, pred, target):
        for i, _theme in enumerate(self.themes):
            if _theme != theme:
                continue
            self.classification_metrics[i].update(pred, target)
    
    def reset(self):
        for _m in self.classification_metrics:
            _m.reset()

    def compute(self, prefix=""):
        res = {}
        for i in range(len(self.themes)):
            _prefix = prefix + self.themes[i] + "_"
            _res = self.classification_metrics[i].compute(prefix=_prefix)
            res.update(_res)
        return res

# + tags=["active-ipynb", "style-student"]
# my_metrics = MultiTheme_Classification_Metrics(
#     themes=["audio", "video"], metrics=["acc", "auc"], num_classes=2
# )
# my_metrics.reset()
# target = torch.tensor([0, 0, 1, 1])
# preds = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]])
# print(my_metrics([preds]*2, [target]*2))
#
# target2 = torch.tensor([0, 1, 1])
# preds2 = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]])
# my_metrics.update([preds2]*2, [target2]*2)
#
# my_metrics.compute(prefix="train_")
