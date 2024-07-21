import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    
    def __init__(self, d_model=2):
        super().__init__()
        
        # initialize up QKV weights, in the original 
        # transformer paper there's no bias added.
        self.w_q = nn.Linear(in_features=d_model,
                             out_features=d_model,
                             bias=False)
        self.w_k = nn.Linear(in_features=d_model,
                             out_features=d_model,
                             bias=False)
        self.w_v = nn.Linear(in_features=d_model,
                             out_features=d_model,
                             bias=False)

        self.row_dim = 0
        self.col_dim = 1
        
    def forward(self, 
                encodings_for_q,
                encodings_for_k,
                encodings_for_v,
                mask=None):

        q = self.w_q(encodings_for_q)
        k = self.w_k(encodings_for_k)
        v = self.w_v(encodings_for_v)
        
        similarity = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_similarity = similarity / torch.tensor(k.size(self.col_dim)**0.5, device=q.device) # square root of d_model
        # print(f"scaled_similarity:\n{scaled_similarity}")

        if mask is not None:
            scaled_similarity = scaled_similarity.masked_fill(mask=mask, value=-1e9)
            # print(f"scaled_similarity with mask:\n{scaled_similarity}")
            """
            tensor([[-1.4174e+00, -1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [-2.1758e-01,  5.7194e-02, -1.0000e+09, -1.0000e+09, -1.0000e+09],
                    [ 1.3718e+00,  2.1296e-01, -1.3261e+00, -1.0000e+09, -1.0000e+09],
                    [-1.1841e-01, -3.9331e-02,  9.9653e-02, -3.1835e-03, -1.0000e+09],
                    [ 9.3959e-01,  9.4240e-02, -9.4471e-01,  9.4343e-02, -6.1551e-01]],
                grad_fn=<MaskedFillBackward0>)
            """

        # dim=0 along row direction (vertical), dim=1 along col direction (horizontal)
        attention_percents = F.softmax(scaled_similarity, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
