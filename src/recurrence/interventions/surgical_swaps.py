"""Sprint S12: Multi-Store Surgical State Swaps Intervention Module.

Enables precise, isolated channel transplantation (RGLRU, Conv1D, KV cache)
between distinct evaluation branches to causally attribute memory retrieval
to specific latent store channels.
"""

from typing import List, Optional, Set, Union
import torch

from recurrence.state.temporal_inventory import RecurrentStateSnapshot


VALID_CHANNELS: Set[str] = {"rglru", "conv", "kv", "k", "v", "all"}


def swap_stores(
    recipient: RecurrentStateSnapshot,
    donor: RecurrentStateSnapshot,
    channels: Union[str, List[str]],
    layers: Optional[List[int]] = None,
) -> RecurrentStateSnapshot:
    """Surgically graft selected channel stores from donor into recipient snapshot.

    Args:
        recipient: The baseline snapshot providing background context.
        donor: The donor snapshot providing the target historical event state.
        channels: One or more channel names from {'rglru', 'conv', 'kv', 'k', 'v', 'all'}.
        layers: Optional subset of layer indices to swap. If None, swaps all shared layers.

    Returns:
        A new RecurrentStateSnapshot containing recipient context with donor channels surgically grafted.
    """
    if isinstance(channels, str):
        channels_list = [channels.lower()]
    else:
        channels_list = [c.lower() for c in channels]

    for c in channels_list:
        if c not in VALID_CHANNELS:
            raise ValueError(f"Invalid channel '{c}'. Must be one of {sorted(VALID_CHANNELS)}")

    # Detached clone of recipient to avoid mutating base trajectory
    grafted = recipient.clone()

    swap_all = "all" in channels_list

    # 1. Surgical RGLRU Swap
    if swap_all or "rglru" in channels_list:
        target_layers = layers if layers is not None else list(recipient.rglru.keys())
        for l in target_layers:
            if l in donor.rglru and l in recipient.rglru:
                assert recipient.rglru[l].shape == donor.rglru[l].shape, (
                    f"RGLRU shape mismatch at layer {l}: {recipient.rglru[l].shape} vs {donor.rglru[l].shape}"
                )
                grafted.rglru[l] = donor.rglru[l].clone()

    # 2. Surgical Conv1D Swap
    if swap_all or "conv" in channels_list:
        target_layers = layers if layers is not None else list(recipient.conv.keys())
        for l in target_layers:
            if l in donor.conv and l in recipient.conv:
                assert recipient.conv[l].shape == donor.conv[l].shape, (
                    f"Conv shape mismatch at layer {l}: {recipient.conv[l].shape} vs {donor.conv[l].shape}"
                )
                grafted.conv[l] = donor.conv[l].clone()

    # 3. Surgical KV Cache Swap
    if swap_all or any(c in channels_list for c in ("kv", "k", "v")):
        target_layers = layers if layers is not None else list(recipient.kv.keys())
        swap_k = swap_all or "kv" in channels_list or "k" in channels_list
        swap_v = swap_all or "kv" in channels_list or "v" in channels_list

        for l in target_layers:
            if l in donor.kv and l in recipient.kv:
                if swap_k and "key" in donor.kv[l] and "key" in recipient.kv[l]:
                    t_rec = recipient.kv[l]["key"]
                    t_don = donor.kv[l]["key"]
                    if isinstance(t_rec, torch.Tensor) and isinstance(t_don, torch.Tensor):
                        assert t_rec.shape == t_don.shape, (
                            f"KV Key shape mismatch at layer {l}: {t_rec.shape} vs {t_don.shape}"
                        )
                        grafted.kv[l]["key"] = t_don.clone()

                if swap_v and "value" in donor.kv[l] and "value" in recipient.kv[l]:
                    t_rec = recipient.kv[l]["value"]
                    t_don = donor.kv[l]["value"]
                    if isinstance(t_rec, torch.Tensor) and isinstance(t_don, torch.Tensor):
                        assert t_rec.shape == t_don.shape, (
                            f"KV Value shape mismatch at layer {l}: {t_rec.shape} vs {t_don.shape}"
                        )
                        grafted.kv[l]["value"] = t_don.clone()

    # Record provenance metadata
    grafted.metadata["swapped_channels"] = channels_list
    grafted.metadata["swap_layers"] = layers if layers is not None else "all"
    return grafted


def add_intervention_matched_noise(
    recipient: RecurrentStateSnapshot,
    donor: RecurrentStateSnapshot,
    channel: str = "rglru",
    seed: int = 42,
) -> RecurrentStateSnapshot:
    """Add random Gaussian noise to recipient matching the layer-wise Frobenius norm of ||donor - recipient||."""
    grafted = recipient.clone()
    gen = torch.Generator().manual_seed(seed)

    if channel == "rglru":
        for l in grafted.rglru:
            if l in donor.rglru:
                t_rec = recipient.rglru[l].float()
                t_don = donor.rglru[l].float()
                diff_frob = float(torch.norm(t_don - t_rec).item())
                if diff_frob > 1e-8:
                    noise = torch.randn(t_rec.shape, generator=gen)
                    noise_frob = float(torch.norm(noise).item())
                    scaled_noise = noise * (diff_frob / (noise_frob + 1e-8))
                    grafted.rglru[l] = (t_rec + scaled_noise.to(t_rec.device)).to(recipient.rglru[l].dtype)
    elif channel == "conv":
        for l in grafted.conv:
            if l in donor.conv:
                t_rec = recipient.conv[l].float()
                t_don = donor.conv[l].float()
                diff_frob = float(torch.norm(t_don - t_rec).item())
                if diff_frob > 1e-8:
                    noise = torch.randn(t_rec.shape, generator=gen)
                    noise_frob = float(torch.norm(noise).item())
                    scaled_noise = noise * (diff_frob / (noise_frob + 1e-8))
                    grafted.conv[l] = (t_rec + scaled_noise.to(t_rec.device)).to(recipient.conv[l].dtype)

    grafted.metadata["noise_perturbed_channel"] = channel
    grafted.metadata["noise_matched_donor"] = True
    grafted.metadata["noise_seed"] = seed
    return grafted

