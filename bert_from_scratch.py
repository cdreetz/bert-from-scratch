import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class CTransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        attn_heads, 
        dropout, 
        bias=True, 
        norm_first=False, 
        layer_norm_eps=1e-5,
        activation=F.relu,
        batch_first=False,
    ):
        super().__init__()
        self.d_model = embed_dim
        self.nhead = attn_heads
        self.dim_feedforward = embed_dim * 4
        self.self_attn = MultiheadAttention(self.d_model, self.nhead, dropout)
        self.linear1 = Linear(self.d_model, self.dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(self.dim_feedforward, self.d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(self.d_model, layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(self.d_model, layer_norm_eps, bias=bias)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_casual=False):
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_casual))
        x = self.norm2(x + self._ff_block(x))

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
        x = self.Linear2(self.dropout(self.activation(self.linear1(x))))
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
        self.tok_embed = CEmbedding(vocab_size, embed_dim)
        self.seg_embed = CEmbedding(n_segments, embed_dim)
        self.pos_embed = CEmbedding(max_len, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)],)

    def forward(self, seq, seg):
        embed_val = self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(self.pos_inp)
        return embed_val


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

    embedding = BERTEmbedding(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, DROPOUT)
    embedding_tensor = embedding(sample_seq, sample_seg)
    print(embedding_tensor.size())

    # bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
    # out = bert(sample_seq, sample_seg)
    # print(out.size())