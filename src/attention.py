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
        
        similarity = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim)) # shape: (seq_len, seq_len)
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
            
            Note1: to easily understsand why masking in this way:
            matmul S x V ensures row-wise(masked to only 1 value, 2 values, 3 values ... n values visible) 
            then dot-product with column-wise(tokens in sequence) so that each token won't attend to future tokens.

            Note2: main difference from inference-time forward():
            1. training forward (aka. Prefill) only once and Q shape (b, s, h) => Result shape (b, s, h) and each h vector represents an "attention" for each input token
            2. inference forward num_new_tokens times and Q shape (b, 1, h) in both prefill and decoding phase (why prefill: we don't compute loss for each token, just the last token in order to generate a new token!)
            3. therefore training don't need KV cache, but inference need it.

            Note2.1: why KV cache still works after layer2 ?:
            1. After prefill phase - each layer caches K, V vectors. It's easy to understand first layer KV cache can be reused because "original tokens did not change" (actual input words).
            2. However, hidden states from 1st layer output -> 2nd input also DO NOT change for previous tokens as well! Why?
                (*TLDR: because of causality so older tokens don't attend to new tokens so their hidden states don't change)
                3. Let's think of decoding phase w/o KV cache, how does it look like using example ["how, "are" -> "you"]:
                4. 1st pass: we encode "how", "are" and generates "you" 
                5. 2nd pass: now we need to encode "how", "are", "you", but as we explained in Note1&2 that "how" and "are" attention won't change at all because they don't attend to new tokens "you"
                6. And when it pass thru MLP block tokens do not attend to each other(MLP is token independent and purely local to each token - just linear transformation along hidden dimension)
                7. Therefore, hidden states will be exactly the same for "how are" part of the sequence between 1st & 2nd pass in 2nd layer, and hence can be *reused* -> hence idea of KV cache. And we just need to compute Q, K, V for new token in each pass.

            Note3: meaning of self-attention
            1. Q technically is always a single token (1, h) - we do (s, h) is just for parallel computing
            2. Use Q as query to dot-product with K, we get the token similarity (or importance) aka. "attention_weights" between this token for every other tokens(include itself!) - intuitively just a list of %percentage
            3. Finally we matmul S with V to get "attention_score" - "vector representation" for the query token, but with dependencies on other tokens' syntax, semantics, or long-range contextual relationships.

            **In short we can simply just rely on input embedding to get an vector representation,
            but that miss a lot of "context" from other tokens in the sentence, hence self-attention essentially add those information by dot-product!
            """

        # dim=0 along row direction (vertical), dim=1 along col direction (horizontal)
        attention_percents = F.softmax(scaled_similarity, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v) # (seq_len, seq_len) x (seq_len, h) == (seq_len, h)
        return attention_scores
