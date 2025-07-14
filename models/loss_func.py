import torch
import torch.nn.functional as F

def compute_cosine_similarity(embeddings, centers):
    """
    Compute cosine similarity between embeddings and cluster centers.
    Args:
        embeddings (Tensor): Item embeddings, shape (num_samples, hidden_units).
        centers (Tensor): Cluster centers, shape (num_clusters, hidden_units).
    Returns:
        cosine_similarities (Tensor): Cosine similarity matrix, shape (num_samples, num_clusters).
    """
    embeddings = F.normalize(embeddings, dim=1)  # Normalize embeddings
    centers = F.normalize(centers, dim=1)  # Normalize cluster centers
    return torch.matmul(embeddings, centers.T)  # Cosine similarity matrix

def compute_entropy_from_similarity(similarity_matrix, temperature=1.0):
    """
    Compute entropy from a similarity matrix using softmax probabilities.
    Args:
        similarity_matrix (Tensor): Cosine similarity matrix, shape (num_samples, num_clusters).
        temperature (float): Temperature parameter for softmax.
    Returns:
        entropy (Tensor): Computed entropy value.
    """
    # Apply temperature scaling and compute softmax probabilities
    probabilities = F.softmax(similarity_matrix * temperature, dim=1)  # Shape: (num_samples, num_clusters)

    # Compute entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1).mean()  # Add epsilon for stability
    return entropy

def alignment_loss(sampled_embeddings, sampled_domains, num_domains, alpha, beta, hidden_units, device, temperature=1.0):
    """
    Compute alignment loss using cosine similarity-based entropy approach.
    Args:
        sampled_embeddings (Tensor): Sampled item embeddings, shape (num_sampled_items, hidden_units).
        sampled_domains (Tensor): Domain IDs for sampled items, shape (num_sampled_items,).
        num_domains (int): Total number of domains.
        alpha (float): Weight for intra-domain entropy loss.
        beta (float): Weight for inter-domain entropy loss.
        hidden_units (int): Dimensionality of the embeddings.
        device: Device to perform the computation on.
        temperature (float): Temperature parameter for softmax.
    Returns:
        alignment_loss (Tensor): Combined alignment loss.
    """
    intra_entropy = 0
    all_embeddings = []
    all_domain_centers = []

    # Compute intra-domain entropy
    for domain_id in range(num_domains):
        mask = (sampled_domains == domain_id)
        if mask.sum() > 0:
            domain_embeddings = sampled_embeddings[mask]
            domain_center = domain_embeddings.mean(dim=0, keepdim=True)  # Compute domain center
            all_embeddings.append(domain_embeddings)
            all_domain_centers.append(domain_center)

            # Compute cosine similarity and entropy for the domain
            similarity_matrix = compute_cosine_similarity(domain_embeddings, domain_center)
            intra_entropy += compute_entropy_from_similarity(similarity_matrix, temperature)

    # Normalize intra-domain entropy
    intra_entropy /= num_domains

    # Combine all embeddings for inter-domain entropy
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_domain_centers = torch.cat(all_domain_centers, dim=0)

    # Compute inter-domain entropy
    similarity_matrix = compute_cosine_similarity(all_embeddings, all_domain_centers)
    inter_entropy = compute_entropy_from_similarity(similarity_matrix, temperature)

    # Combine intra-domain and inter-domain entropy into the alignment loss
    return alpha * intra_entropy - beta * inter_entropy



def compute_diversity_entropy(embeddings, temperature=1.0):
    """
    Compute diversity entropy for embeddings using pairwise cosine similarity.
    Args:
        embeddings (Tensor): Item embeddings, shape (num_samples, hidden_units).
        temperature (float): Temperature parameter for softmax.
    Returns:
        entropy (Tensor): Diversity entropy value.
    """
    # Compute pairwise cosine similarity
    similarity_matrix = compute_cosine_similarity(embeddings, embeddings)  # Shape: (N, N)

    # Apply temperature scaling and compute softmax probabilities
    probabilities = F.softmax(similarity_matrix * temperature, dim=1)  # Shape: (N, N)

    # Compute entropy for each row
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1).mean()
    return entropy

def compute_inter_domain_entropy(sampled_embeddings, sampled_domains, domain_centers, temperature=1.0):
    """
    Compute inter-domain entropy excluding contribution of embeddings from their own domain centers.
    Args:
        sampled_embeddings (Tensor): Sampled embeddings, shape (num_samples, hidden_units).
        sampled_domains (Tensor): Domain IDs for sampled items, shape (num_samples,).
        domain_centers (Tensor): Domain centers, shape (num_domains, hidden_units).
        temperature (float): Temperature parameter for softmax.
    Returns:
        entropy (Tensor): Computed entropy value for inter-domain compactness.
    """
    # Compute cosine similarity between sampled embeddings and domain centers
    similarity_matrix = compute_cosine_similarity(sampled_embeddings, domain_centers)  # Shape: (num_samples, num_domains)

    # Mask out similarities to the same domain center
    mask = torch.zeros_like(similarity_matrix, device=sampled_embeddings.device)
    for i, domain_id in enumerate(sampled_domains):
        mask[i, domain_id] = 1  # Mark same-domain center

    # Exclude same-domain similarities by setting them to a very low value
    similarity_matrix = similarity_matrix.masked_fill(mask.bool(), -float("inf"))

    # Apply temperature scaling and compute softmax probabilities
    probabilities = F.softmax(similarity_matrix * temperature, dim=1)  # Shape: (num_samples, num_domains)

    # Compute entropy for each sampled embedding and average
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1).mean()
    return entropy


def alignment_loss_with_sampled_entropy(
    sampled_embeddings, sampled_domains, num_domains, alpha_base,  temperature=1.0
):
    """
    Compute alignment loss with relative scaling of alpha and beta based on intra-domain and inter-domain scales.
    Args:
        sampled_embeddings (Tensor): Sampled item embeddings, shape (num_samples, hidden_units).
        sampled_domains (Tensor): Domain IDs for sampled items, shape (num_samples,).
        num_domains (int): Total number of domains (|E|).
        alpha_base (float): Base value for alpha (controls intra-domain term weight).
        hidden_units (int): Dimensionality of the embeddings.
        device: Device to perform the computation on.
        temperature (float): Temperature parameter for softmax.
    Returns:
        alignment_loss (Tensor): Combined alignment loss.
    """
    num_samples = sampled_embeddings.size(0)  # Total number of embeddings (N)

    # Set alpha and beta based on the relative scaling
    alpha = alpha_base
    beta = alpha_base * (num_samples / (num_domains**3))  # Adjust beta relative to N and |E|

    # Compute intra-domain diversity
    intra_diversity = 0
    domain_centers = []
    for domain_id in range(num_domains):
        mask = (sampled_domains == domain_id)
        if mask.sum() > 0:
            domain_embeddings = sampled_embeddings[mask]
            domain_centers.append(domain_embeddings.mean(dim=0, keepdim=True))  # Compute domain center

            # Compute diversity entropy for intra-domain loss
            diversity_entropy = compute_diversity_entropy(domain_embeddings, temperature)
            intra_diversity += diversity_entropy
    intra_diversity /= num_domains  # Average across domains

    # Compute inter-domain compactness via entropy
    domain_centers = torch.cat(domain_centers, dim=0)  # Shape: (num_domains, hidden_units)
    inter_entropy = compute_inter_domain_entropy(
        sampled_embeddings, sampled_domains, domain_centers, temperature
    )

    # Combine intra-domain and inter-domain terms
    return - alpha * intra_diversity + beta * inter_entropy

