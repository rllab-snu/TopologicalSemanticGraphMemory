import torch
import torch.nn as nn
import math


class CategoryEncoding(nn.Module):
    def __init__(self, n_filters=32, max_len=80):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(CategoryEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        ce = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        ce[:, 0::2] = torch.sin(position * div_term)
        ce[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('ce', ce)  # buffer is a tensor, not a variable, (L, D)

    def forward(self, categories):
        """
        :Input: (*, L, D)
        :Output: (*, L, D) the same size as input
        """
        ce = []
        for b in range(categories.shape[0]):
            ce.append(self.ce.data[categories[b].long()])  # (#x.size(-2), n_filters)
        ce_tensor = torch.stack(ce)
        # x = x + ce_tensor
        return ce_tensor

