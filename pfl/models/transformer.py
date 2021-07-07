import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu

class TransformerBlock(nn.Module):
    def __init__(self, num_attn_heads, input_dim, attn_hidden_dim, fc_hidden_dim, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.k = attn_hidden_dim
        self.num_heads = num_attn_heads

        self.wq = nn.Linear(input_dim, num_attn_heads * attn_hidden_dim, bias=False)
        self.wk = nn.Linear(input_dim, num_attn_heads * attn_hidden_dim, bias=False)
        self.wv = nn.Linear(input_dim, num_attn_heads * attn_hidden_dim, bias=False)
        self.wc = nn.Linear(num_attn_heads * attn_hidden_dim, input_dim, bias=False)
        self.dropout_attn = nn.Dropout(dropout)

        self.w1 = nn.Linear(input_dim, fc_hidden_dim)
        self.dropoutfc = nn.Dropout(dropout)
        self.w2 = nn.Linear(fc_hidden_dim, input_dim)

        self.layernorm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout)

        # print('Using custom initialization!')
        nn.init.normal_(self.wq.weight, 0, .02)
        nn.init.normal_(self.wk.weight, 0, .02)
        nn.init.normal_(self.wv.weight, 0, .02)
        nn.init.normal_(self.wc.weight, 0, .02)

        nn.init.normal_(self.w1.weight, 0, .02)
        nn.init.constant_(self.w1.bias, 0.0)
        nn.init.normal_(self.w2.weight, 0, .02)
        nn.init.constant_(self.w2.bias, 0.0)

    def forward(self, x, mask):
        # x: (seq_len, B, d); mask: (seq_len, seq_len)
        # mask is 0 or -inf. Add to pre-softmax scores
        seq_len, batch_size, embed_dim = x.shape
        query = self.wq(x).contiguous().view(seq_len, batch_size * self.num_heads, self.k).transpose(0, 1)  # (seq_len, B*H, k)
        key  = self.wk(x).contiguous().view(seq_len, batch_size * self.num_heads, self.k).transpose(0, 1)  # (seq_len, B*H, k)
        value = self.wv(x).contiguous().view(seq_len, batch_size * self.num_heads, self.k).transpose(0, 1) # (seq_len, B*H, k)
        alpha = torch.bmm(query, key.transpose(1, 2)) + mask  # (seq_len, B*H, B*H)
        alpha = softmax(alpha / math.sqrt(self.k), dim=-1)  # (seq_len, B*H, B*H)
        alpha = self.dropout_attn(alpha)  # (seq_len, B*H, B*H)
        u = torch.bmm(alpha, value)  # (seq_len, B*H, k)
        u = u.transpose(0, 1).contiguous().view(seq_len, batch_size, self.num_heads*self.k)   # (seq_len, B, H*k)
        u = self.layernorm1(x + self.dropout1(self.wc(u)))  # (seq_len, B, d)
        z = self.w2(self.dropoutfc(relu(self.w1(u))))  # (seq_len, B, d)
        z = self.layernorm2(u + self.dropout2(z))
        return z  # (seq_len, B, d)

class Transformer(nn.Module):
    def __init__(self, seq_len, vocab_size, input_dim, attn_hidden_dim, fc_hidden_dim,
                 num_attn_heads, num_layers, tied_weights=False, dropout_tr=0., dropout_io=0.
    ):
        super(Transformer, self).__init__()
        print(f"""Constructing a transformer model with:
                    - seq_len = {seq_len}
                    - vocab_size = {vocab_size}
                    - input_dim = {input_dim}
                    - attn_hidden_dim = {attn_hidden_dim}
                    - fc_hidden_dim = {fc_hidden_dim}
                    - num_attn_heads = {num_attn_heads}
                    - num_layers = {num_layers}
                    - tied_weights = {tied_weights}
                    - dropout_tr = {dropout_tr}
                    - dropout_io = {dropout_io}
            """)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mask = None
        self.pos = None
        self.dims = input_dim
        self.tied_weights = tied_weights
        self.dropout = dropout_tr

        self.positional_embedding = nn.Embedding(seq_len, input_dim)
        self.drop_i = nn.Dropout(dropout_io)
        self.word_embedding = nn.Embedding(vocab_size, input_dim)
        self.transformer = nn.ModuleList()
        for i in range(num_layers):
            self.transformer.append(TransformerBlock(num_attn_heads, input_dim, attn_hidden_dim, fc_hidden_dim, dropout_tr))

        if not tied_weights: self.decoder = nn.Linear(input_dim, vocab_size)
        self.drop_o = nn.Dropout(dropout_io)
        self.bias = nn.Parameter(torch.ones(vocab_size))

        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.word_embedding.weight, 0, .02)
        if not self.tied_weights: nn.init.normal_(self.decoder.weight, 0, .02)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        # x: (seq_len, batch_size)
        if self.mask is None or self.mask.shape[0] != x.shape[0]:
            self.mask = torch.triu(torch.ones(len(x), len(x)))
            self.mask.masked_fill_(self.mask == 0, float('-inf')).masked_fill_(self.mask == 1, float(0.0))
            self.mask = self.mask.transpose(0, 1).to(x.device)
            self.pos = torch.arange(0, x.shape[0], dtype=torch.long).to(x.device)

        x = self.word_embedding(x) * math.sqrt(self.dims)
        p = self.positional_embedding(self.pos)[:, None, :]
        z = relu(self.drop_i(x) + self.drop_i(p))
        for layer in self.transformer:
            z = layer(z, self.mask)

        z = self.drop_o(z)
        outputs = torch.matmul(z, self.word_embedding.weight.t()) if self.tied_weights else self.decoder(z)
        # return log_softmax(outputs + self.bias, dim=-1)
        return outputs + self.bias  # pre-softmax weights

