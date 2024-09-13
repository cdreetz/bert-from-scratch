import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules import ModuleList
from torch.nn import Linear, LayerNorm, Dropout

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _no_grad_fill_(tensor, val):
    with torch.no_grad():
        return tensor.fill_(val)

def constant_(tensor, val):
    return _no_grad_fill_(tensor, val)

def _no_grad_uniform_(tensor, a, b, generator=None):
    with torch.no_grad():
        return tensor.uniform_(a, b, generator=None)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def cxavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -a, a)

class CNonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)

class CMultiheadAttention(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        attn_heads, 
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = attn_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // self.num_heads

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim),)
            )
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim),)
            )
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim),)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim),)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim),)

        self.out_proj = CNonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias,
        )
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim),))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim),))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            cxavier_uniform_(self.in_proj_weight)
        else:
            cxavier_uniform_(self.q_proj_weight)
            cxavier_uniform_(self.k_proj_weight)
            cxavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        
    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_casual=False
    ):

        why_not_fast_path = ""
        is_batched = query.dim() == 3

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            # is_casual=is_casual
        )

        return attn_output, attn_output_weights
    
        
        

class CTransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward = 2048,
        dropout=0.1, 
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False, 
        bias=True, 
        device=None,
        dtype=None,

    ):
        super().__init__()
        self.self_attn = CMultiheadAttention(
           d_model,
           nhead,
           dropout=dropout,
           bias=bias,
           batch_first=batch_first,
        )
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.relu

    def forward(
        self, 
        src,
        src_mask=None, 
        src_key_padding_mask=None, 
        is_casual=False
    ):
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_casual))
        x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask, is_casual=False):
        x = self.self_attn(x, x, x, 
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask, 
            need_weights=False,
            is_casual=is_casual
        )[0]
        return self.dropout1(x)

    # feed-forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class CTransformerEncoder(nn.Module):
    def __init__(
        self, 
        encoder_layer: "CTransformerEncoderLayer", 
        num_layers, 
        enable_nested_tensor=True, 
        mask_check=True
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        self.enable_nested_tensor = enable_nested_tensor
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        enc_layer = "encoder_layer"

    def forward(self, src, mask=None, src_key_padding_mask=None, is_casual=None):
        output = src

        for mod in self.layers:
            output = mod(output)

        return output


class CEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, _weight=None, _freeze=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(
            torch.empty(num_embeddings, embedding_dim), 
            requires_grad=not _freeze
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self):
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(input, self.weight)

    @classmethod
    def from_pretrained(cls, embeddings):
        assert (embeddings.dim() == 2), "Embeddings must be 2D"
        rows, cols = embeddings.shape
        embedding = cls(num_embeddings=rows, embedding_dim=cols)
        return embedding


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.seg_embed = nn.Embedding(n_segments, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)],)

    def forward(self, seq, seg):
        embed_val = self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(self.pos_inp)
        embed_val = self.drop(embed_val)
        return embed_val


class CBERTEmbedding(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, dropout):
        super().__init__()
        self.tok_embed = CEmbedding(vocab_size, embed_dim)
        self.seg_embed = CEmbedding(n_segments, embed_dim)
        self.pos_embed = CEmbedding(max_len, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)],)

    def forward(self, seq, seg):
        embed_val = self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(self.pos_inp)
        embed_val = self.drop(embed_val)
        return embed_val


class CBERT(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, n_layers, attn_heads, dropout):
        super().__init__()
        self.embedding = CBERTEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
        self.encoder_layer = CTransformerEncoderLayer(embed_dim, attn_heads, embed_dim*4)
        self.encoder_block = CTransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.encoder_block(out)
        return out

class BERT(nn.Module):
    def __init__(self, vocab_size, n_segments, max_len, embed_dim, n_layers, attn_heads, dropout):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, attn_heads, embed_dim*4)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)

    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.encoder_block(out)
        return out

if __name__ == "__main__":
    VOCAB_SIZE = 30000
    N_SEGMENTS = 3       # A and B plus padding
    MAX_LEN = 512
    EMBED_DIM = 768
    N_LAYERS = 12
    ATTN_HEADS = 12
    DROPOUT = 0.1

    sample_seq = torch.randint(high=VOCAB_SIZE, size=[MAX_LEN,])
    sample_seg = torch.randint(high=N_SEGMENTS, size=[MAX_LEN,])

    cembedding = CBERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    embedding_tensor = embedding(sample_seq, sample_seg)
    cembedding_tensor = cembedding(sample_seq, sample_seg)
    print(cembedding_tensor.size())
    print(embedding_tensor.size())

    bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
    out = bert(sample_seq, sample_seg)
    print(out.size())

    cbert = CBERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
    out = cbert(sample_seq, sample_seg)
    print(out.size())

