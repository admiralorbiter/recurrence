"""Orthogonalized monitor/content intervention operator for Gate A."""

from typing import Tuple
import torch


def compute_orthogonalized_direction(
    monitor_vec: torch.Tensor,
    content_vec: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """Computes m_perp = m - proj_c(m), normalized to unit norm.
    
    Args:
        monitor_vec: Raw candidate monitor vector (D,).
        content_vec: First-order content direction vector (D,).
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        Orthogonalized unit vector m_perp (D,).
    """
    m = monitor_vec.flatten().to(torch.float32)
    c = content_vec.flatten().to(torch.float32)
    
    c_norm_sq = torch.dot(c, c)
    if c_norm_sq < eps:
        # If content vector is negligible, return normalized monitor vector
        m_norm = torch.norm(m)
        return m / (m_norm + eps)
        
    proj_c_m = (torch.dot(m, c) / c_norm_sq) * c
    m_perp = m - proj_c_m
    m_perp_norm = torch.norm(m_perp)
    
    if m_perp_norm < eps:
        raise ValueError("Monitor vector is collinear with content vector; cannot orthogonalize.")
        
    return m_perp / m_perp_norm


def apply_dose_intervention(
    hidden_state: torch.Tensor,
    direction: torch.Tensor,
    dose_lambda: float,
    scale_sigma: float = 1.0
) -> torch.Tensor:
    """Applies a scaled directional perturbation: h_new = h + lambda * sigma * direction."""
    orig_shape = hidden_state.shape
    orig_dtype = hidden_state.dtype
    
    h_flat = hidden_state.flatten().to(torch.float32)
    d_flat = direction.flatten().to(torch.float32)
    
    # Ensure unit norm for direction
    d_unit = d_flat / (torch.norm(d_flat) + 1e-8)
    
    delta = dose_lambda * scale_sigma * d_unit
    h_intervened = h_flat + delta
    
    return h_intervened.view(orig_shape).to(orig_dtype)
