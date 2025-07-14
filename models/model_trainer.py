import torch
import numpy as np
from collections import Counter
import random
from models.loss_func import  alignment_loss_with_sampled_entropy
import torch

def train_model(model, dataloader, optimizer, num_epochs, num_items, early_stop_patience, model_save_path, device):
    """
    Train the model using BPR loss with negative sampling.
    Args:
        model: The recommendation model.
        dataloader: DataLoader for the training dataset.
        optimizer: Optimizer for updating model weights.
        num_epochs: Number of training epochs.
        num_items: Total number of items.
        early_stop_patience: Patience for early stopping.
        model_save_path: Path to save the best model.
        device: Device to run the training on.
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            train_seq, val_item, _ = batch  # train_seq: (batch_size, seq_len), val_item: (batch_size)
            train_seq, val_item = train_seq.to(device), val_item.to(device)

            # Forward pass: logits shape (batch_size, num_items)
            logits = model(train_seq)

            # Negative sampling: sample 5 negative items per positive item
            neg_items = torch.randint(1, num_items + 1, (val_item.size(0), 5), device=device)

            # Positive logits: get logits for ground-truth items
            pos_logits = logits.gather(1, val_item.unsqueeze(1)).squeeze(1)  # Shape: (batch_size)

            # Negative logits: get logits for sampled negative items
            neg_logits = logits.gather(1, neg_items)  # Shape: (batch_size, 5)

            # Repeat positive logits to match the number of negative samples
            pos_logits = pos_logits.unsqueeze(1).repeat(1, neg_items.size(1))  # Shape: (batch_size, 5)

            # Flatten both for BPR loss calculation
            pos_logits = pos_logits.view(-1)
            neg_logits = neg_logits.view(-1)

            # Compute BPR loss
            loss = bpr_loss(pos_logits, neg_logits)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered. No improvement in the last {early_stop_patience} epochs.")
            break

    print(f"Training completed. Best model saved to {model_save_path}")




def resample_embeddings(sampled_embeddings, sampled_domains, batch_size, num_domains, device):
    """
    Dynamically resample embeddings from sampled domains to match the batch size.
    Args:
        sampled_embeddings (Tensor): Sampled item embeddings, shape (num_sampled_items, embedding_dim).
        sampled_domains (Tensor): Domain IDs for sampled items, shape (num_sampled_items,).
        batch_size (int): Number of embeddings to sample per domain.
        num_domains (int): Total number of domains (excluding the current domain).
        device: Device to perform computation on.
    Returns:
        resampled_embeddings (Tensor): Resampled embeddings, shape (num_domains * batch_size, embedding_dim).
        resampled_domains (Tensor): Resampled domain IDs, shape (num_domains * batch_size,).
    """
    resampled_embeddings = []
    resampled_domains = []

    for domain_id in range(num_domains):
        mask = (sampled_domains == domain_id)
        domain_embeddings = sampled_embeddings[mask]

        # Randomly sample batch_size embeddings for the domain
        if domain_embeddings.size(0) > 0:
            indices = torch.randperm(domain_embeddings.size(0))[:batch_size]
            resampled_embeddings.append(domain_embeddings[indices])
            resampled_domains.append(torch.full((batch_size,), domain_id, dtype=torch.long, device=device))

    resampled_embeddings = torch.cat(resampled_embeddings, dim=0)
    resampled_domains = torch.cat(resampled_domains, dim=0)

    return resampled_embeddings, resampled_domains

def train_model_with_alignment(
    model, dataloader, optimizer, num_epochs, num_items, num_aux_domains, sampled_domains, sampled_embeddings,
    alpha,  early_stop_patience, model_save_path, device
):
    """
    Train the model using BPR loss combined with alignment loss.
    Args:
        model: The recommendation model.
        dataloader: DataLoader for the training dataset.
        optimizer: Optimizer for updating model weights.
        num_epochs: Number of training epochs.
        num_items: Total number of items.
        num_aux_domains: Total number of domains (including the current domain).
        sampled_domains: Tensor of domain IDs for sampled embeddings, shape (num_sampled_items,).
        sampled_embeddings: Tensor of sampled embeddings, shape (num_sampled_items, embedding_dim).
        alpha: Weight for intra-domain alignment loss.
        beta: Weight for inter-domain alignment loss.
        early_stop_patience: Patience for early stopping.
        model_save_path: Path to save the best model.
        device: Device to run the training on.
    """
    # Move sampled domains and embeddings to the device
    sampled_domains = sampled_domains.to(device)
    sampled_embeddings = sampled_embeddings.to(device)

    # Current domain ID is distinct from sampled domain IDs
    current_domain_id = num_aux_domains

    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_rec_loss = 0.0
        total_align_loss = 0.0
        for batch in dataloader:
            train_seq, val_item, _ = batch  # train_seq: (batch_size, seq_len), val_item: (batch_size)
            train_seq, val_item = train_seq.to(device), val_item.to(device)

            # Resample embeddings for the current domain
            current_embeddings = model.sample_internal_embeddings(sample_size=train_seq.size(0), device=device)

            # Resample embeddings for auxiliary domains
            batch_size = train_seq.size(0)  # Match the batch size
            resampled_embeddings, resampled_domains = resample_embeddings(
                sampled_embeddings, sampled_domains, batch_size, num_aux_domains , device
            )

            # Project all embeddings
            projected_current_embeddings = model.irm_projection_embeddings(current_embeddings)  # Shape: (batch_size, hidden_units)
            projected_resampled_embeddings = model.irm_projection_embeddings(resampled_embeddings)  # Shape: (batch_size * num_domains, hidden_units)

            # Assign domain IDs
            current_domains = torch.full((projected_current_embeddings.size(0),), current_domain_id, dtype=torch.long, device=device)

            # Combine all embeddings and domains
            combined_embeddings = torch.cat([projected_current_embeddings, projected_resampled_embeddings], dim=0)
            combined_domains = torch.cat([current_domains, resampled_domains], dim=0)

            # Forward pass: logits shape (batch_size, num_items)
            logits = model(train_seq)

            # Negative sampling: sample 5 negative items per positive item
            neg_items = torch.randint(1, num_items + 1, (val_item.size(0), 5), device=device)

            # Positive logits: get logits for ground-truth items
            pos_logits = logits.gather(1, val_item.unsqueeze(1)).squeeze(1)  # Shape: (batch_size)

            # Negative logits: get logits for sampled negative items
            neg_logits = logits.gather(1, neg_items)  # Shape: (batch_size, 5)

            # Repeat positive logits to match the number of negative samples
            pos_logits = pos_logits.unsqueeze(1).repeat(1, neg_items.size(1))  # Shape: (batch_size, 5)

            # Flatten both for BPR loss calculation
            pos_logits = pos_logits.view(-1)
            neg_logits = neg_logits.view(-1)

            # Compute BPR loss
            rec_loss = bpr_loss(pos_logits, neg_logits)

            # Compute alignment loss
            align_loss = alignment_loss_with_sampled_entropy(
                combined_embeddings, combined_domains, num_aux_domains + 1, alpha,
            )

            # Total loss
            total_loss_batch = rec_loss + align_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()
            total_rec_loss += rec_loss.item()
            total_align_loss += align_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, BPR Loss: {total_rec_loss/len(dataloader):.4f}, Alignment Loss: {total_align_loss/len(dataloader):.4f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered. No improvement in the last {early_stop_patience} epochs.")
            break

    print(f"Training completed. Best model saved to {model_save_path}")


def evaluate_model_with_neg_sampling(model, dataloader, top_k_set, num_items, device, num_negatives=100):
    """
    Evaluate the model performance using negative sampling, compatible with MostPop and sequential models.

    Args:
        model: The trained recommendation model.
        dataloader: DataLoader providing (train_seq, val_item, test_item).
        top_k_set: List of top-K values for which metrics are calculated.
        num_items: Total number of items in the dataset.
        device: The device on which the model is evaluated (CPU or GPU).
        num_negatives: Number of negative samples to use for evaluation.
    Returns:
        recall_sum: Dictionary containing recall sums for each k in top_k_set.
        ndcg_sum: Dictionary containing NDCG sums for each k in top_k_set.
        total: Total number of items evaluated.
    """
    model.eval()
    with torch.no_grad():
        recall_sum = {k: 0.0 for k in top_k_set}
        ndcg_sum = {k: 0.0 for k in top_k_set}
        total = 0

        all_items = torch.arange(1, num_items + 1, device=device)  # Items from 1 to num_items

        for batch in dataloader:
            train_seq, _, test_item = batch  # Use test_item as ground truth
            train_seq, test_item = train_seq.to(device), test_item.to(device)

            batch_size = test_item.size(0)

            for i in range(batch_size):
                # Interacted items for the user
                interacted_items = set(train_seq[i].tolist())
                interacted_items.discard(0)  # Remove padding index

                # Sample negative items
                negative_items = list(set(all_items.tolist()) - interacted_items)
                negative_samples = random.sample(negative_items, min(num_negatives, len(negative_items)))

                # Add ground truth to negatives
                candidates = torch.tensor(negative_samples + [test_item[i].item()], device=device).unsqueeze(0)

                # Check if the model has a `predict` method that supports candidates (for mostpop only)
                if hasattr(model, "recommend"):
                    scores = model.recommend(candidates)
                else:
                    # Sequential models: compute logits for all items, then filter candidates
                    eval_seq = train_seq[i].unsqueeze(0)
                    all_logits = model.predict(eval_seq)
                    scores = all_logits[:, candidates.squeeze(0)]

                # Rank the candidate items
                _, top_indices = torch.topk(scores, max(top_k_set), dim=-1)
                top_k_items = candidates[0, top_indices.squeeze(0)]

                # Compute metrics
                ground_truth = test_item[i].item()
                for k in top_k_set:
                    recall_sum[k] += recall_at_k(top_k_items[:k], ground_truth, k)
                    ndcg_sum[k] += ndcg_at_k(top_k_items[:k], ground_truth, k)

            total += batch_size

        # Compute average metrics
        for k in top_k_set:
            avg_recall = recall_sum[k] / total * 100
            avg_ndcg = ndcg_sum[k] / total * 100
            print(f"Test Recall@{k}: {avg_recall:.4f}%, NDCG@{k}: {avg_ndcg:.4f}%")

        return recall_sum, ndcg_sum, total


def evaluate_model(model, dataloader, top_k_set, device):
    """
    Evaluate the model performance by calculating recall and NDCG.

    Args:
        model: The trained recommendation model.
        dataloader: DataLoader containing the evaluation dataset.
        top_k_set: List of top-K values for which metrics are calculated.
        device: The device on which the model is evaluated (CPU or GPU).
    Returns:
        recall_sum: Dictionary containing recall sums for each k in top_k_set.
        ndcg_sum: Dictionary containing NDCG sums for each k in top_k_set.
        total: Total number of items evaluated.
    """
    model.eval()
    with torch.no_grad():
        recall_sum = {k: 0.0 for k in top_k_set}
        ndcg_sum = {k: 0.0 for k in top_k_set}
        total = 0
        for batch in dataloader:
            train_seq, val_item, test_item = batch
            train_seq, val_item, test_item = train_seq.to(device), val_item.to(device), test_item.to(device)

            eval_seq = torch.cat([train_seq, val_item.unsqueeze(1)], dim=1)
            logits = model.predict(eval_seq)
            _, top_k_items = torch.topk(logits, max(top_k_set), dim=-1)

            for i in range(test_item.size(0)):
                for k in top_k_set:
                    recall_sum[k] += recall_at_k(top_k_items[i][:k], test_item[i].item(), k)
                    ndcg_sum[k] += ndcg_at_k(top_k_items[i][:k], test_item[i].item(), k)
            total += test_item.size(0)

        for k in top_k_set:
            avg_recall = recall_sum[k] / total * 100
            avg_ndcg = ndcg_sum[k] / total * 100
            print(f"Test Recall@{k}: {avg_recall:.4f}%, NDCG@{k}: {avg_ndcg:.4f}%")

        return recall_sum, ndcg_sum, total




# BPR Loss function
def bpr_loss(pos_logits, neg_logits):
    return -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-10))

# Metrics: Recall and NDCG@K
def recall_at_k(pred_items, ground_truth, k):
    return torch.tensor(ground_truth in pred_items[:k], dtype=torch.float32).item()

def ndcg_at_k(pred_items, ground_truth, k):
    if ground_truth in pred_items[:k]:
        rank = pred_items[:k].tolist().index(ground_truth) + 1
        return (1 / np.log2(rank + 1)).item()
    else:
        return 0.0

def evaluate_most_pop(model, dataloader, top_k_list):
    """
    Evaluate the MostPop model using Recall@K and NDCG@K.
    :param model: MostPop model instance.
    :param dataloader: DataLoader providing (train_seq, val_item, test_item).
    :param top_k_list: List of top-k values to compute metrics for.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    recall_sum = {k: 0 for k in top_k_list}
    ndcg_sum = {k: 0 for k in top_k_list}
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            train_seq, _, test_item = batch  # Use test_item for evaluation
            train_seq = train_seq.to(device)
            test_item = test_item.to(device)

            # Predict item popularity scores
            predictions = model.predict(train_seq)
            top_k_predictions = predictions.argsort(dim=-1, descending=True)  # Shape: (batch_size, num_items)

            for i in range(train_seq.size(0)):  # Iterate over batch
                ground_truth = test_item[i].item()
                pred_items = top_k_predictions[i].tolist()  # Get predictions for this user

                for k in top_k_list:
                    recall_sum[k] += recall_at_k(pred_items, ground_truth, k)
                    ndcg_sum[k] += ndcg_at_k(pred_items, ground_truth, k)

            total += train_seq.size(0)

    # Compute average metrics
    for k in top_k_list:
        recall = recall_sum[k] / total * 100  # Convert to percentage
        ndcg = ndcg_sum[k] / total
        print(f"Recall@{k}: {recall:.4f}%, NDCG@{k}: {ndcg:.4f}")

    return recall_sum, ndcg_sum


def fit_most_pop(model, dataloader):
    """
    Fit the MostPop model using the training DataLoader.
    :param model: MostPop model instance.
    :param dataloader: DataLoader providing (train_seq, val_item, test_item).
    """
    print("Fitting MostPop model...")
    item_counts = Counter()

    for batch in dataloader:
        train_seq, _, _ = batch  # Only use train_seq for counting item frequencies
        flat_items = train_seq.flatten().tolist()  # Flatten the sequences
        item_counts.update(flat_items)

    # Normalize counts to compute item popularity
    model.popularity = torch.zeros(model.num_items + 1)  # +1 for padding
    for item, count in item_counts.items():
        model.popularity[item] = count
    model.popularity[0] = 0  # remove the recommendation probability for padding index
    model.popularity /= model.popularity.sum()  # Normalize to probabilities

    print("MostPop model fitted.")