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
class PositionEmbedding(nn.Module):

    MODE_EXPAND = "MODE_EXPAND"
    MODE_ADD = "MODE_ADD"
    MODE_CONCAT = "MODE_CONCAT"

    def __init__(self, num_embeddings, embedding_dim, mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(
                torch.Tensor(num_embeddings * 2 + 1, embedding_dim)
            )
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()
        # print("PositionEmbedding, weight shape is ", self.weight.shape)

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = (
                torch.clamp(x, -self.num_embeddings, self.num_embeddings)
                + self.num_embeddings
            )
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        # print(x.shape, seq_len, self.num_embeddings, self.embedding_dim)
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError("Unknown mode: %s" % self.mode)

    def extra_repr(self):
        return "num_embeddings={}, embedding_dim={}, mode={}".format(
            self.num_embeddings,
            self.embedding_dim,
            self.mode,
        )
