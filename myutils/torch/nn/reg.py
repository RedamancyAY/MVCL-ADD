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
import torch
import torch.nn as nn


# + tags=[]
class Regularization(nn.Module):
    def __init__(self, weight_decay, p=2):
        """
        Args:
            model: 模型
            weight_decay: 正则化参数
            p: 范数计算中的幂指数值，默认求2范数,当p=0为L2正则化,p=1为L1正则化
        """
        super().__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.weight_decay = weight_decay
        self.p = p
        

    def forward(self, model):
        weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(
            weight_list, self.weight_decay, p=self.p
        )
        return reg_loss

    def get_weight(self, model):
        """获得模型的权重列表
        Args:
            model: 模型
        """
        weight_list = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        """计算张量范数
        Args: 
            weight_list: 参数列表
            p: 范数计算中的幂指数值，默认求2范数
            weight_decay: 正则化参数
        """
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.linalg.vector_norm(w, ord=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, model):
        """打印权重列表信息
        Args:
            weight_list: 参数列表
        """
        weight_list = self.get_weight(model)
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
