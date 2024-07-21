import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.optim import Adam
from attention import Attention
from position_encoder import PositionEncoding


# we only need to inherit LightningModule once from the top parent class
# TODO: handle batching
class DecoderOnlyTransformer(L.LightningModule):

    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__()

        # given a token id (indice) retrieve the trained word embeddings (a vector)
        self.we = nn.Embedding(num_embeddings=num_tokens,
                               embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model,
                                   max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        
        self.loss = nn.CrossEntropyLoss() # applies softmax internally

    def forward(self, token_ids):

        wte = self.we(token_ids)
        wtpe = self.pe(wte)

        """
        tril == "tri lower"

        tensor([[1,  0,  0,  0,  0],
                [1,  1,  0,  0,  0],
                [1,  1,  1,  0,  0],
                [1,  1,  1,  1,  0],
                [1,  1,  1,  1,  1]])
        """
        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)))).to(token_ids.device)
        mask = mask == 0
        """
        tensor([[False,  True,  True,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False, False, False,  True],
                [False, False, False, False, False]])
        """

        # print(f"mask:\n{mask}")

        self_attention_values = self.self_attention(wtpe, wtpe, wtpe, mask=mask)
        residual_connection_values = wtpe + self_attention_values
        fc_layer_output = self.fc_layer(residual_connection_values)

        # print(f"fc_layer_output:\n{fc_layer_output}")
        """
        fc_layer_output:
        tensor([[ -2.6690,  11.8545,   5.2197, -12.1674,  -8.9135,   4.5789],
                [  5.8160,   5.4494,  10.0095,  -4.0716, -19.0141,  -5.3181],
                [  6.7615, -21.6996,  -8.6329,  12.5679,   5.7718,  -1.1031],
                [ -1.7717,   1.4681,  -1.3168,  -5.5312,  -0.4650,   5.4854],
                [ -8.3536, -10.9095, -17.6151,  -1.4847,  24.3724,  16.6519]],
            grad_fn=<AddmmBackward0>)
        """
        return fc_layer_output

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)
     
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])
        return loss
