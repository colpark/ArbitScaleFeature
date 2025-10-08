# CORRECTED approximate_relative_distances function for LAINRDecoder
# Replace the existing function in Cell 4 of cifar10_experiments_CORRECTED.ipynb

def approximate_relative_distances(self, target_index, H, W, m):
    """
    Compute spatial bias based on distance

    Args:
        target_index: (HW,) - patch indices for each query pixel
        H, W: patch grid dimensions (e.g., 16×16)
        m: number of LP tokens (e.g., 256)

    Returns:
        rel_distances: (m, HW) - bias matrix [LP_tokens × pixels]
    """
    alpha = self.alpha
    N = H * W  # Number of patches = 256

    # Normalize patch indices to [0, 1]
    t = target_index.float() / N  # (HW,) where HW could be 1024 pixels

    # LP token positions (evenly distributed in [0, 1])
    token_positions = torch.tensor(
        [(i + 0.5) / m for i in range(m)],
        device=target_index.device
    )  # (m,) = (256,)

    # Broadcast to create (m, HW) matrix
    t_expanded = t.unsqueeze(0)  # (1, HW) = (1, 1024)
    tokens_expanded = token_positions.unsqueeze(1)  # (m, 1) = (256, 1)

    # Distance-based bias: -alpha * |distance|^2
    # Result shape: (m, HW) = (256, 1024)
    rel_distances = -alpha * torch.abs(t_expanded - tokens_expanded)**2

    return rel_distances


# USAGE in forward():
# This will create bias of shape (256, 1024) which transposes to (1024, 256)
# Which matches the attention shape (B, H, HW, L) = (B, H, 1024, 256)
