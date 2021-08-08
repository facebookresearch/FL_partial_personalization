from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu
from torchinfo import summary

from .base_model import PFLBaseModel

class TransformerBlock(nn.Module):
    def __init__(self, num_attn_heads, input_dim, attn_hidden_dim, fc_hidden_dim, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.input_dim = input_dim
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

        self.use_adapter = False
        self.adapter1 = False
        self.adapter2 = False

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
        # Apply attention
        alpha = torch.bmm(query, key.transpose(1, 2)) + mask  # (seq_len, B*H, B*H)
        alpha = softmax(alpha / math.sqrt(self.k), dim=-1)  # (seq_len, B*H, B*H)
        alpha = self.dropout_attn(alpha)  # (seq_len, B*H, B*H)
        u = torch.bmm(alpha, value)  # (seq_len, B*H, k)
        u = u.transpose(0, 1).contiguous().view(seq_len, batch_size, self.num_heads*self.k)   # (seq_len, B, H*k)
        # Apply first FC (post-attention)
        u = self.dropout1(self.wc(u))  # (seq_len, B, d)
        # Apply adapter if necessary
        if self.use_adapter:
            u = self.adapter1(u)  # (seq_len, B, d)
        # Apply skip connection
        u = x + u  # (seq_len, B, d)
        # Apply layer norm
        u = self.layernorm1(u)  # (seq_len, B, d)
        # Apply FC x 2
        z = self.dropout2(self.w2(self.dropoutfc(relu(self.w1(u)))))  # (seq_len, B, d)
        # Apply adapter if necessary
        if self.use_adapter:
            u = self.adapter2(u)  # (seq_len, B, d)
        # Apply skip connection
        z = u + z  # (seq_len, B, d)
        # Apply layer norm
        z = self.layernorm2(z)
        return z  # (seq_len, B, d)

    def add_adapters(self, adapter_hidden_dim):
        if not self.use_adapter:
            self.use_adapter = True
            self.adapter1 = AdapterBlock(self.input_dim, adapter_hidden_dim)
            self.adapter2 = AdapterBlock(self.input_dim, adapter_hidden_dim)

class AdapterBlock(nn.Module):
    def __init__(self, input_dim, adapter_hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, adapter_hidden_dim)
        self.linear2 = nn.Linear(adapter_hidden_dim, input_dim)
        # initialize weights to a small constant
        for module in [self.linear1, self.linear2]:
            nn.init.normal_(module.weight, 0, .01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x): # x: (seq_len, B, d)
        # down-project
        u = relu(self.linear1(x))  # (seq_len, B, h)
        # up-project
        u = self.linear2(u)  # (seq_len, B, d)
        # skip connection
        u = x + u
        return u


class Transformer(PFLBaseModel):
    def __init__(self, seq_len, vocab_size, input_dim, attn_hidden_dim, fc_hidden_dim,
                 num_attn_heads, num_layers, tied_weights=False, dropout_tr=0., dropout_io=0.,
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

        if not tied_weights:  # output layer
            self.decoder = nn.Linear(input_dim, vocab_size, bias=False)  # bias vector below
        self.drop_o = nn.Dropout(dropout_io)
        self.bias = nn.Parameter(torch.ones(vocab_size))

        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.word_embedding.weight, 0, .02)
        if not self.tied_weights: nn.init.normal_(self.decoder.weight, 0, .02)
        nn.init.constant_(self.bias, 0.0)
        self.is_on_client = None
        self.is_on_server = None

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
        return outputs + self.bias  # pre-softmax weights

    def print_summary(self, train_batch_size):
        device = next(self.parameters()).device
        print(summary(self, input_size=(self.seq_len, train_batch_size), 
                      dtypes=[torch.int64], device=device))

    def load_pretrained_and_prepare(self, state_dict, train_mode, layers_to_finetune, adapter_hidden_dim):
        """ Load state_dict, initialize finetune modules and set requires_grad to freeze weights.
        """
        assert train_mode in ['train', 'finetune', 'finetune_tr_layer', 
                              'finetune_inp_layer', 'finetune_out_layer',
                              'adapter', 'prefix']
        # load state_dict 
        self.load_state_dict(state_dict, strict=False)

        # Untie weights for IO layers if required
        if self.tied_weights and train_mode in ['finetune_inp_layer', 'finetune_out_layer']:
            self.tied_weights = False
            self.decoder = nn.Linear(*self.word_embedding.weight.t().shape, bias=False) # hidden dim -> vocab size
            with torch.no_grad():  # initialize with word embedding
                self.decoder.weight.copy_(self.word_embedding.weight)
            if train_mode in ['finetune_inp_layer']:  
                # finetune only the word embedding
                self.decoder.weight.requires_grad_(False)
                self.word_embedding.weight.requires_grad_(True)
            elif train_mode in ['finetune_out_layer']:
                # finetune only the linear layer
                self.decoder.weight.requires_grad_(True)
                self.word_embedding.weight.requires_grad_(False)

        if layers_to_finetune is None:  # do not fine tune
            layers_to_finetune = []
        if train_mode == 'finetune_tr_layer' and len(layers_to_finetune) is None:
            raise ValueError(f'No transformer layers to finetune. Nothing to do')

        # Set requires_grad based on `train_mode`
        if train_mode in ['train', 'finetune']:  # all parameters are to be tuned
            def do_finetune(name):
                return True
        elif 'finetune_tr_layer' in train_mode:
            # Finetune a specific transformer layer
            def do_finetune(name):
                return any([f'transformer.{i}' in name for i in layers_to_finetune])
        elif train_mode in ['finetune_inp_layer']:
            # Fine tune positional and word embeddings
            def do_finetune(name):
                return ('embedding' in name)
        elif train_mode in ['finetune_out_layer']:
            # Fine tune final linear layer
            def do_finetune(name):
                return ('bias' == name) or ('decoder' in name)
        elif train_mode in ['adapter']:
            # Train adapter modules (+ batch norm)
            def do_finetune(name):
                return ('adapter' in name) or ('layernorm' in name)
            # Add adapter modules
            for block in self.transformer:
                block.add_adapters(adapter_hidden_dim)
        else:
            raise ValueError(f'Unknown train_mode: {train_mode}')

        # set requires_grad for those parameters which need to be modified
        for name, param in self.named_parameters():
            param.requires_grad_(do_finetune(name))
    
    def split_server_and_client_params(self, client_mode, layers_to_client, adapter_hidden_dim):
        """ Initialize adapter modules if necessary and split parameters into server_parameters and client_parameters.
        """
        device = next(self.parameters()).device
        if self.is_on_client is not None:
            raise ValueError('This model has already been split across clients and server.')
        assert client_mode in ['none', 'tr_layer', 'inp_layer', 'out_layer', 'adapter', 'prefix', 'interpolate']
        is_on_server = None

        # Untie weights for IO layers if required
        if self.tied_weights and client_mode in ['inp_layer', 'out_layer']:
            self.tied_weights = False
            self.decoder = nn.Linear(*self.word_embedding.weight.t().shape, bias=False) # hidden dim -> vocab size
            with torch.no_grad():  # initialize with word embedding
                self.decoder.weight.copy_(self.word_embedding.weight)

        if layers_to_client is None:  # do not fine tune
            layers_to_client = []
        if client_mode == 'tr_layer' and len(layers_to_client) is None:
            raise ValueError(f'No transformer layers to client. Choose fedavg setting')

        # Set requires_grad based on `client_mode`
        if client_mode == 'none' or client_mode is None:  # the entire model is on the server
            is_on_client = lambda _: False
            # is_on_server = lambda _: True
        elif 'tr_layer' in client_mode:
            # Send a specific transformer layer to client
            def is_on_client(name):
                return any([f'transformer.{i}' in name for i in layers_to_client])
        elif client_mode in ['inp_layer']:
            # Send positional and word embeddings to clients
            def is_on_client(name):
                return ('embedding' in name)
        elif client_mode in ['finetune_out_layer']:
            # Send final linear layer to client
            def is_on_client(name):
                return ('bias' == name) or ('decoder' in name)
        elif client_mode in ['adapter']:
            # Send adapter modules (+ normalization) to clients
            def is_on_client(name):
                return ('adapter' in name) or ('layernorm' in name)
            # Add adapter modules
            for block in self.transformer:
                block.add_adapters(adapter_hidden_dim)
        elif client_mode == 'interpolate':
            is_on_client = lambda _: True
            is_on_server = lambda _: True
        else:
            raise ValueError(f'Unknown client_mode: {client_mode}')
        if is_on_server is None:
            def is_on_server(name):
                return not is_on_client(name)
        self.is_on_client = is_on_client
        self.is_on_server = is_on_server
        self.to(device)