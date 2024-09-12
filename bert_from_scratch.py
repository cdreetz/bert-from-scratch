import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

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