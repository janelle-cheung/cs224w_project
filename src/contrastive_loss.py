"""
Contrastive Loss for EEG embeddings.

Implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
for learning embeddings where epochs from the same patient are pulled together
and epochs from different patients are pushed apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.

    Used for contrastive learning where:
    - Positive pairs: Epochs from the same patient
    - Negative pairs: Epochs from different patients

    Args:
        temperature: Temperature scaling parameter (default: 0.07)
        use_cosine_similarity: Use cosine similarity instead of dot product (default: True)
    """

    def __init__(self, temperature: float = 0.07, use_cosine_similarity: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, embeddings: torch.Tensor, sleep_stages: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of embeddings.

        Args:
            embeddings: (batch_size, embedding_dim) tensor of embeddings
            sleep_stages: (batch_size,) tensor of sleep stage labels (0-4)

        Returns:
            Scalar loss value
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Normalize embeddings if using cosine similarity
        if self.use_cosine_similarity:
            embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix: (batch_size, batch_size)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same sleep stage)
        positive_mask = self._create_positive_mask(sleep_stages, device)

        # Create mask for negative pairs
        negative_mask = self._create_negative_mask(batch_size, device)

        # For numerical stability, subtract max from similarity matrix
        similarity_matrix = similarity_matrix - torch.max(similarity_matrix, dim=1, keepdim=True)[0].detach()

        # Compute denominator: sum of exp(sim) for all negative pairs + exp(sim) for positive pairs
        # exp_sim: (batch_size, batch_size)
        exp_sim = torch.exp(similarity_matrix) * negative_mask

        # Sum over negatives for each anchor
        denominator = exp_sim.sum(dim=1, keepdim=True)

        # For each anchor, compute loss w.r.t. all its positives
        # Numerator: exp(sim) for positive pairs
        numerator = torch.exp(similarity_matrix) * positive_mask

        # Compute log probability for each positive pair
        # log_prob: (batch_size, batch_size)
        log_prob = torch.log(numerator / (denominator + 1e-8) + 1e-8)

        # Mask to select only positive pairs and compute mean
        num_positives_per_row = positive_mask.sum(dim=1)

        # Handle case where some rows have no positives
        valid_rows = num_positives_per_row > 0

        if not valid_rows.any():
            # No valid positive pairs in batch
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute loss: negative mean log probability of positive pairs
        loss = -(log_prob * positive_mask).sum(dim=1) / (num_positives_per_row + 1e-8)

        # Average over valid rows
        loss = loss[valid_rows].mean()
        print(f"NT-Xent Loss: {loss.item():.4f}")

        return loss

    def _create_positive_mask(self, sleep_stages: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create mask for positive pairs (same sleep stage, excluding self).

        Args:
            sleep_stages: (batch_size,) tensor of sleep stage labels
            device: Device to create tensor on

        Returns:
            (batch_size, batch_size) boolean mask
        """
        batch_size = sleep_stages.shape[0]
        mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and sleep_stages[i] == sleep_stages[j]:
                    mask[i, j] = True

        return mask.float()

    def _create_negative_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create mask for negative pairs (all pairs excluding self).

        Args:
            batch_size: Size of batch
            device: Device to create tensor on

        Returns:
            (batch_size, batch_size) boolean mask
        """
        mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=device)
        mask.fill_diagonal_(False)
        return mask.float()

class HybridContrastiveLoss(nn.Module):
    """
    Hybrid loss combining contrastive loss and classification loss.

    Useful for semi-supervised or multi-task learning where we want to:
    1. Learn patient-specific representations (contrastive)
    2. Learn sleep stage classification (supervised)

    Args:
        temperature: Temperature for NT-Xent loss (default: 0.07)
        alpha: Weight for contrastive loss (default: 0.5)
        beta: Weight for classification loss (default: 0.5)
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.contrastive_loss = NTXentLoss(temperature=temperature)
        self.classification_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        embeddings: torch.Tensor,
        patient_ids: List[str],
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute hybrid loss.

        Args:
            embeddings: (batch_size, embedding_dim) embeddings
            patient_ids: List of patient IDs
            logits: (batch_size, num_classes) classification logits
            labels: (batch_size,) ground truth labels

        Returns:
            (total_loss, contrastive_loss, classification_loss)
        """
        contrastive = self.contrastive_loss(embeddings, patient_ids)
        classification = self.classification_loss(logits, labels)

        total_loss = self.alpha * contrastive + self.beta * classification

        return total_loss, contrastive, classification


def gather_positive_negative_indices(patient_ids: List[str], anchor_idx: int) -> Tuple[List[int], List[int]]:
    """
    Helper function to gather indices of positive and negative examples for a given anchor.

    Args:
        patient_ids: List of patient IDs for the batch
        anchor_idx: Index of the anchor sample

    Returns:
        (positive_indices, negative_indices) tuple of index lists
    """
    anchor_patient = patient_ids[anchor_idx]

    positive_indices = []
    negative_indices = []

    for idx, patient_id in enumerate(patient_ids):
        if idx == anchor_idx:
            continue  # Skip self

        if patient_id == anchor_patient:
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)

    return positive_indices, negative_indices


def compute_positive_negative_stats(patient_ids: List[str]) -> dict:
    """
    Compute statistics about positive and negative pairs in a batch.

    Useful for monitoring batch composition during training.

    Args:
        patient_ids: List of patient IDs for the batch

    Returns:
        Dictionary with statistics
    """
    batch_size = len(patient_ids)
    unique_patients = len(set(patient_ids))

    total_positives = 0
    total_negatives = 0
    min_positives = float('inf')
    max_positives = 0

    for i in range(batch_size):
        positives, negatives = gather_positive_negative_indices(patient_ids, i)
        n_pos = len(positives)
        n_neg = len(negatives)

        total_positives += n_pos
        total_negatives += n_neg
        min_positives = min(min_positives, n_pos)
        max_positives = max(max_positives, n_pos)

    stats = {
        'batch_size': batch_size,
        'unique_patients': unique_patients,
        'total_positive_pairs': total_positives,
        'total_negative_pairs': total_negatives,
        'avg_positives_per_sample': total_positives / batch_size,
        'avg_negatives_per_sample': total_negatives / batch_size,
        'min_positives_per_sample': min_positives if min_positives != float('inf') else 0,
        'max_positives_per_sample': max_positives,
    }

    return stats


if __name__ == '__main__':
    """Test the contrastive loss."""
    print("Testing NT-Xent Loss\n" + "=" * 50)

    # Create dummy data
    batch_size = 8
    embedding_dim = 64

    embeddings = torch.randn(batch_size, embedding_dim)
    patient_ids = ['P01', 'P01', 'P02', 'P02', 'P03', 'P01', 'P04', 'P02']

    print(f"Batch size: {batch_size}")
    print(f"Patient IDs: {patient_ids}")
    print(f"Embeddings shape: {embeddings.shape}\n")

    # Compute batch statistics
    stats = compute_positive_negative_stats(patient_ids)
    print("Batch Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test NT-Xent loss
    print("\n" + "=" * 50)
    print("Testing NT-Xent Loss")
    print("=" * 50)

    loss_fn = NTXentLoss(temperature=0.07)
    loss = loss_fn(embeddings, patient_ids)

    print(f"Loss: {loss.item():.4f}")

    # Test with different temperatures
    print("\nLoss with different temperatures:")
    for temp in [0.01, 0.05, 0.07, 0.1, 0.5, 1.0]:
        loss_fn = NTXentLoss(temperature=temp)
        loss = loss_fn(embeddings, patient_ids)
        print(f"  Temperature={temp:.2f}: Loss={loss.item():.4f}")

    # Test hybrid loss
    print("\n" + "=" * 50)
    print("Testing Hybrid Loss")
    print("=" * 50)

    logits = torch.randn(batch_size, 5)  # 5 sleep stages
    labels = torch.randint(0, 5, (batch_size,))

    hybrid_loss_fn = HybridContrastiveLoss(temperature=0.07, alpha=0.5, beta=0.5)
    total_loss, contrastive_loss, classification_loss = hybrid_loss_fn(
        embeddings, patient_ids, logits, labels
    )

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Contrastive Loss: {contrastive_loss.item():.4f}")
    print(f"Classification Loss: {classification_loss.item():.4f}")

    # Test positive/negative gathering
    print("\n" + "=" * 50)
    print("Testing Positive/Negative Gathering")
    print("=" * 50)

    for anchor_idx in range(min(3, batch_size)):
        pos_indices, neg_indices = gather_positive_negative_indices(patient_ids, anchor_idx)
        print(f"\nAnchor {anchor_idx} (Patient {patient_ids[anchor_idx]}):")
        print(f"  Positive indices: {pos_indices}")
        print(f"  Negative indices: {neg_indices}")
