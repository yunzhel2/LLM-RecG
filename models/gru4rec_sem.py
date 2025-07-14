import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4Rec(nn.Module):
    def __init__(self, hidden_units, num_layers, dropout_rate, pretrained_item_embeddings=None):
        """
        GRU4Rec with dynamic pretrained embeddings for zero-shot inference.
        Args:
            hidden_units (int): Hidden size for the GRU and embeddings.
            num_layers (int): Number of GRU layers.
            dropout_rate (float): Dropout rate.
            pretrained_item_embeddings (Tensor, optional): Pretrained item embeddings.
        """
        super(GRU4Rec, self).__init__()

        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        if pretrained_item_embeddings is not None:
            # Use pretrained embeddings and add a projection layer if dimensions don't match
            self.pretrained_dim = pretrained_item_embeddings.shape[1]
            self.pretrained_item_embedding = nn.Embedding.from_pretrained(
                pretrained_item_embeddings, freeze=True, padding_idx=0
            )
            self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            print(f"Using pretrained embeddings with projection layer ({self.pretrained_dim} -> {hidden_units}).")
        else:
            # Placeholder for dynamically loaded embeddings
            self.pretrained_item_embedding = None
            self.projection_layer = None

        # GRU layers for user sequence representation
        self.gru = nn.GRU(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, item_seq):
        """
        Forward pass to compute logits for all items using dot product similarity.
        Args:
            item_seq (Tensor): Input item sequences, shape (batch_size, seq_len).
        Returns:
            logits (Tensor): Logits for all items, shape (batch_size, num_items).
        """
        device = item_seq.device

        # Use pretrained embeddings if available
        if self.pretrained_item_embedding is not None:
            pretrained_emb = self.pretrained_item_embedding(item_seq).to(device)
            item_emb = self.projection_layer(pretrained_emb)  # Project to hidden_units
        else:
            raise ValueError("No pretrained embeddings loaded. Use `load_new_pretrain_embeddings` first.")

        # Apply dropout
        item_emb = self.dropout(item_emb)

        # Pass through GRU layers to get user representation
        _, user_rep = self.gru(item_emb)  # user_rep: (num_layers, batch_size, hidden_units)
        user_rep = user_rep[-1]  # Take the last GRU layer's output, shape: (batch_size, hidden_units)

        # Use dynamically loaded embeddings
        item_embeddings = self.projection_layer(self.pretrained_item_embedding.weight)  # Shape: (num_items, hidden_units)

        # Compute dot product similarity
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