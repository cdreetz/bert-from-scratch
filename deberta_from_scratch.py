class CDeBERTaMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # content projections
        self.qc_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kc_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # position projections
        self.qp_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kp_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, position_embeddings, attention_mask=None):
        batch_size, seq_len, _ = query.size()

        # compute content and positional projections
        qc = self.qc_proj(query)
        kc = self.kc_proj(key)
        qp = self.qp_proj(position_embeddings)
        kp = self.kp_proj(position_embeddings)
        v = self.v_proj(value)

        # reshape for multi-head attention
        qc = qc.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kc = kc.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        qp = qp.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kp = kp.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # compute attention scores
        # content-to-content
        ac = torch.matmul(qc, kc.transpose(-1, -2))
        # content-to-positional
        bd = torch.matmul(qc, kp.transpose(-1, -2))
        # positional-to-content
        ef = torch.matmul(qp, kc.transpose(-1, -2))
        # positional-to-positional
        gh = torch.matmul(qp, kp.transpose(-1, -2))

        # combine attention scores 
        attention_scores = ac + bd + ef + gh

        # scale attention scores
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # apply attention mask (if any)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # compute attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = nn.Dropout(self.dropout)(attention_probs)

        # compute context 
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.out_proj(context)
        return output



class CDeBERTaTransformerEncoderLayer(nn.Module):
    def __init__(self,d_model, nhead, dim_feedforward = 2048, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = CDeBERTaMultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, position_embeddings, attention_mask=None):
        # self-attention with disentangled mechanism
        src2 = self.self_attn(src, src, src, position_embeddings, attention_mask)
        src = src + src2
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + src2
        src = self.norm2(src)
        
        return src

class CDeBERTaEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_position_embeddings, n_segments, dropout):
        super().__init__()
        self.content_embedding = CEmbedding(vocab_size, embed_dim)
        self.position_embeddings = CEmbedding(max_position_embeddings, embed_dim)
        self.token_type_embeddings = CEmbedding(n_segments, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        content_embeddings = self.content_embedding(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = content_embeddings + position_embeddings
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class CDeBERTa(nn.Module):
    def __init__(self, vocab_size, max_position_embeddings, embed_dim, n_segments, n_layers, n_heads, dropout):
        super().__init__()
        self.embedding = CDeBERTaEmbedding(vocab_size, embed_dim, max_position_embeddings, n_segments, dropout)
        self.layers = nn.ModuleList([
            CDeBERTaTransformerEncoderLayer(embed_dim, n_heads, embed_dim*4, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # generate position IDs
        position_ids = create_position_ids_from_input_ids(input_ids, padding_idx=0)

        # get embeddings
        embedding_output = self.embedding(input_ids, position_ids)

        position_embeddings = self.embedding.position_embeddings(position_ids)

        if token_type_ids is not None:
            token_type_embeddings = self.embedding.token_type_embeddings(token_type_ids)
            embedding_output += token_type_embeddings

        # apply transformer layers
        hidden_states = embedding_output
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)

        # final layer norm
        hidden_states = self.norm(hidden_states)
        print(f"Final output: {hidden_states.shape}")
        
        return hidden_states



if __name__ == "__main__":
    VOCAB_SIZE = 30522
    MAX_POSITION_EMBEDDINGS = 512
    EMBED_DIM = 768
    N_SEGMENTS = 3
    N_LAYERS = 12
    N_HEADS = 12
    DROPOUT = 0.1

    batch_size = 8
    seq_len = 512 # would this work with 128?
    
    # the following hyperparameters are per the paper
    # batch size = 2k
    # sequence length = 512
    # embedding dimension = 768
    # number of attention heads = 12
    # number of layers = 12
    # dropout = 0.1
    # attn head size = 64
    # feed forward dim = 3072
    # learning rate = 2e-4

    # trained on DGX-2 x4 (64 v100 16gb gpus) for 10 days
    # lambda labs $35.20/hr (64 x v100) for 10 days = $8448

    # memory used 64 16gb gpus = 1024gb

    # at h100's 80gb mem, only 12.8 or 16 gpus are needed
    # at a100's 80gb mem, only 12.8 or 16 gpus are needed
    # ends up $28.64/hr for 10 days = $6873.60
    # but i would assume it would train faster 
    # apparently they are ~3.4x faster
    # so $28.64/hr for 3.5 days = $2405.76

    # but h100 is supposedly 2-9x faster than a100
    # so 3.5 days down to ~1day at $47.84/hr is $1148.16


    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    attention_mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
    token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long)

    deberta = CDeBERTa(VOCAB_SIZE, MAX_POSITION_EMBEDDINGS, EMBED_DIM, N_SEGMENTS, N_LAYERS, N_HEADS, DROPOUT)
    out = deberta(input_ids, attention_mask, token_type_ids)
    print(out.shape)




