import torch
import torch.nn as nn
import numpy as np


class SASRec(nn.Module):
    def __init__(self, hidden_units, max_seq_length, num_heads, num_layers, dropout_rate, pretrained_item_embeddings=None):
        """
        SASRec with dynamic pretrained embeddings for zero-shot inference.
        Args:
            hidden_units (int): Dimension of the hidden units.
            max_seq_length (int): Maximum sequence length.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dropout_rate (float): Dropout rate.
            pretrained_item_embeddings (Tensor, optional): Pretrained item embeddings.
        """
        super(SASRec, self).__init__()

        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Use pretrained embeddings or define placeholders
        if pretrained_item_embeddings is not None:
            self.pretrained_dim = pretrained_item_embeddings.shape[1]
            self.pretrained_item_embedding = nn.Embedding.from_pretrained(
                pretrained_item_embeddings, freeze=True, padding_idx=0
            )
            self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            print(f"Using pretrained embeddings with projection layer ({self.pretrained_dim} -> {hidden_units}).")
        else:
            self.pretrained_item_embedding = None
            self.projection_layer = None

        # Position embeddings
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_units)

        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer Encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=hidden_units,
                dropout=dropout_rate
            ) for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)

    def forward(self, item_seq):
        """
        Forward pass to compute logits for all items using dot product similarity.
        Args:
            item_seq (Tensor): Input item sequences, shape (batch_size, seq_len).
        Returns:
            logits (Tensor): Logits for all items, shape (batch_size, num_items).
        """
        device = item_seq.device
        batch_size, seq_len = item_seq.size()

        # Use pretrained embeddings
        if self.pretrained_item_embedding is not None:
            pretrained_emb = self.pretrained_item_embedding(item_seq).to(device)
            item_emb = self.projection_layer(pretrained_emb)  # Shape: (batch_size, seq_len, hidden_units)
        else:
            raise ValueError("No pretrained embeddings loaded. Use `load_new_pretrain_embeddings` first.")

        # Add position embeddings
        position_indices = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(position_indices)  # Shape: (batch_size, seq_len, hidden_units)
        seq_emb = item_emb + pos_emb

        # Apply dropout
        seq_emb = self.dropout(seq_emb)

        # Generate attention mask (causal mask to prevent attending to future tokens)
        attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        src_key_padding_mask = (item_seq == 0)  # Mask padding items

        # Pass through transformer layers
        for attn_layer in self.attention_layers:
            seq_emb = attn_layer(
                seq_emb.permute(1, 0, 2),  # Shape: (seq_len, batch_size, hidden_units)
                src_key_padding_mask=src_key_padding_mask
            ).permute(1, 0, 2)

        # Apply layer normalization
        user_rep = self.layer_norm(seq_emb[:, -1, :])  # Use the representation of the last item in the sequence

        # Compute logits using dot product with item embeddings
        item_embeddings = self.projection_layer(self.pretrained_item_embedding.weight)  # Shape: (num_items, hidden_units)
        logits = torch.matmul(user_rep, item_embeddings.T)  # Shape: (batch_size, num_items)

        return logits

    def predict(self, item_seq, candidate_items=None):
        """
        Predict scores for candidate items or all items using dot product similarity.
        Args:
            item_seq (Tensor): Input item sequences, shape (batch_size, seq_len).
            candidate_items (Tensor, optional): Candidate items for ranking, shape (batch_size, num_candidates).
                                               If None, scores for all items are returned.
        Returns:
            scores (Tensor): Scores for candidate items or all items.
        """
        logits = self.forward(item_seq)  # Get logits for all items
        if candidate_items is not None:
            # Gather scores for candidate items only
            scores = torch.gather(logits, dim=1, index=candidate_items)
            return scores
        return logits

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings):
        """
        Load new pretrained item embeddings dynamically.
        Args:
            pretrained_item_embeddings (Tensor): New pretrained item embeddings.
        """
        device = next(self.parameters()).device  # Get the current model's device
        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.to(device), freeze=True, padding_idx=0
        )
        print(f"New pretrained embeddings loaded successfully on device: {device}")


    def projection_external_embeddings(self, item_embeddings):
        if self.pretrained_item_embedding is None:
            raise ValueError(
                f"No projection layer included in current model.")

        return self.projection_layer(item_embeddings)