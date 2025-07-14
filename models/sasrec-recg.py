import torch
import torch.nn as nn

class SASRecWithDomainAlignment(nn.Module):
    """
    SASRec with dynamic pretrained embeddings and domain alignment projections.
    """
    def __init__(
        self,
        hidden_units: int,
        max_seq_length: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pretrained_item_embeddings: torch.Tensor = None
    ):
        super(SASRecWithDomainAlignment, self).__init__()

        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Pretrained item embeddings + projections
        if pretrained_item_embeddings is not None:
            self.pretrained_dim = pretrained_item_embeddings.shape[1]
            self.pretrained_item_embedding = nn.Embedding.from_pretrained(
                pretrained_item_embeddings, freeze=True, padding_idx=0
            )
            # projection for recommendation
            self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            # projection for domain alignment
            self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            # merge back to hidden_units
            self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)
        else:
            self.pretrained_item_embedding = None
            self.projection_layer = None
            self.domain_alignment_projection_layer = None
            self.merge_layer = None

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=hidden_units,
                dropout=dropout_rate
            ) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)

    def forward(self, item_seq: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass computing logits for all items.
        """
        device = item_seq.device
        batch_size, seq_len = item_seq.size()

        if self.pretrained_item_embedding is None:
            raise ValueError("No pretrained embeddings loaded. Call load_new_pretrain_embeddings first.")

        # lookup and project embeddings
        pretrained_emb = self.pretrained_item_embedding(item_seq)  # (B, L, D)
        emb_rec = self.projection_layer(pretrained_emb)           # (B, L, H)
        emb_irm = self.domain_alignment_projection_layer(pretrained_emb)  # (B, L, H)

        # dropout
        emb_rec = self.dropout(emb_rec)
        emb_irm = self.dropout(emb_irm)

        # merge
        merged = torch.cat((emb_rec, emb_irm), dim=-1)           # (B, L, 2H)
        seq_emb = self.merge_layer(merged)                        # (B, L, H)

        # add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_emb = seq_emb + self.pos_embedding(positions)
        seq_emb = self.dropout(seq_emb)

        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        padding_mask = (item_seq == 0)

        # transformer layers
        out = seq_emb
        for layer in self.attention_layers:
            out = layer(out.permute(1, 0, 2), src_key_padding_mask=padding_mask).permute(1, 0, 2)

        user_rep = self.layer_norm(out[:, -1, :])  # (B, H)

        # final logits
        # reuse projection layer on embedding weights
        all_item_emb = self.projection_layer(self.pretrained_item_embedding.weight)  # (N, H)
        logits = torch.matmul(user_rep, all_item_emb.T)                           # (B, N)

        return logits

    def predict(
        self,
        item_seq: torch.LongTensor,
        candidate_items: torch.LongTensor = None
    ) -> torch.Tensor:
        logits = self.forward(item_seq)
        if candidate_items is not None:
            return torch.gather(logits, dim=1, index=candidate_items)
        return logits

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings: torch.Tensor):
        """
        Dynamically load new pretrained embeddings.
        """
        device = next(self.parameters()).device
        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.to(device), freeze=True, padding_idx=0
        )
        # update dims if needed
        self.projection_layer = nn.Linear(pretrained_item_embeddings.size(1), self.hidden_units).to(device)
        self.domain_alignment_projection_layer = nn.Linear(pretrained_item_embeddings.size(1), self.hidden_units).to(device)
        self.merge_layer = nn.Linear(self.hidden_units * 2, self.hidden_units).to(device)

    def projection_embeddings(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        if self.projection_layer is None:
            raise ValueError("Projection layer not initialized.")
        return self.projection_layer(item_embeddings)

    def irm_projection_embeddings(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        if self.domain_alignment_projection_layer is None:
            raise ValueError("IRM projection layer not initialized.")
        return self.domain_alignment_projection_layer(item_embeddings)

    def sample_internal_embeddings(
        self, sample_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.pretrained_item_embedding is None:
            raise ValueError("No internal embeddings available.")
        num_items = self.pretrained_item_embedding.num_embeddings
        idx = torch.randint(0, num_items, (sample_size,), device=device)
        return self.pretrained_item_embedding(idx)
