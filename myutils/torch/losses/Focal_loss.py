# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# This criterion is a implemenation of Focal Loss, which is 
# proposed in Focal Loss for Dense Object Detection.
# $$
# \mathrm{FL}\left(\mathrm{p}_{\mathrm{t}}\right)=-\alpha_{\mathrm{t}}\left(1-\mathrm{p}_{\mathrm{t}}\right)^\gamma \log \left(\mathrm{p}_{\mathrm{t}}\right)
# $$
# The losses are averaged across observations for each minibatch.

class Focal_loss(nn.Module):
    r"""
    This criterion is a implemenation of Focal Loss, which is proposed in
    Focal Loss for Dense Object Detection.

        Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

    The losses are averaged across observations for each minibatch.

    Args:
        class_num(int): the number of classes in classification
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed
            examples (p > .5), putting more focus on hard, misclassiﬁed examples
        size_average(bool): By default, the losses are averaged over observations for
            each minibatch. However, if the field size_average is set to False, the
            losses are instead summed for each minibatch.
    """

    def __init__(self, class_num=2, alpha=None, alpha_trainable=False, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        if alpha_trainable:
            self.alpha = Variable(self.alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

# + tags=["style-student", "active-ipynb"]
# SEED = 42
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
#
# inputs = torch.rand(5, 2)
# targets = torch.tensor([0, 0, 1, 1, 0], dtype=torch.int64)
# loss = Focal_loss()
# print(loss(inputs, targets))
