import torch
import torch.nn as nn


class PositionEncoding(nn.Module):

    def __init__(self, d_model=2, max_len=6):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        # unsqueeze(1) turns a sequence of numbers into column matrix
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index/d_model)

        # pe[:, 0::2] shape (6,1)
        # pe[:, 0] shape (1,6)
        # the goal is to assign the 1st column with sin, and 2nd with cos so both should
        # work depending on the shape of torch.sin/cos()
        # here we use pe[:, 0::2] because var position is in shape (6,1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # used for register a buffer which is not a model parameter
        # and will be moved to GPU if we use one.
        self.register_buffer('pe', pe)

    def forward(self, wte):
        return wte + self.pe[:wte.size(0), :] # can be simplified to self.pe[:wte.size(0)]
