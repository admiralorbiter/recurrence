"""Layer-Indexed Multi-Store Temporal Inventory for Recurrent State Models (Sprint S10).

Provides explicit, inspectable, layer-indexed data structures to snapshot, serialize,
clone, swap, zero, and restore the 3 distinct temporal stores of hybrid recurrent models:
1. RGLRU Recurrent States (rglru[layer_idx] -> Tensor)
2. 1D Temporal Convolution Buffers (conv[layer_idx] -> Tensor)
3. Local Attention Sliding KV Cache (kv[layer_idx] -> {key: Tensor, value: Tensor})
"""

from dataclasses import dataclass, field
import io
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass
class RecurrentStateSnapshot:
    """Composite layer-indexed temporal state snapshot for hybrid recurrent models."""
    rglru: Dict[int, torch.Tensor] = field(default_factory=dict)
    conv: Dict[int, torch.Tensor] = field(default_factory=dict)
    kv: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    cache_position: int = 0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self, device: Optional[Union[str, torch.device]] = None) -> "RecurrentStateSnapshot":
        """Return a detached, deep memory clone of this snapshot on the target device."""
        target_device = torch.device(device) if device is not None else self.device

        cloned_rglru = {}
        for l_idx, tensor in self.rglru.items():
            t = tensor.detach().clone()
            if target_device is not None:
                t = t.to(target_device)
            cloned_rglru[l_idx] = t

        cloned_conv = {}
        for l_idx, tensor in self.conv.items():
            t = tensor.detach().clone()
            if target_device is not None:
                t = t.to(target_device)
            cloned_conv[l_idx] = t

        cloned_kv = {}
        for l_idx, kv_dict in self.kv.items():
            cloned_kv[l_idx] = {}
            for k, tensor in kv_dict.items():
                t = tensor.detach().clone()
                if target_device is not None:
                    t = t.to(target_device)
                cloned_kv[l_idx][k] = t

        return RecurrentStateSnapshot(
            rglru=cloned_rglru,
            conv=cloned_conv,
            kv=cloned_kv,
            cache_position=self.cache_position,
            device=target_device if target_device is not None else self.device,
            dtype=self.dtype,
            metadata=dict(self.metadata),
        )

    def zero_store(
        self,
        store_type: Literal["rglru", "conv", "kv", "all"],
        layer_idx: Optional[int] = None,
    ) -> None:
        """Surgically zero specified temporal store in-place without modifying other stores."""
        if store_type in ("rglru", "all"):
            layers = [layer_idx] if layer_idx is not None else list(self.rglru.keys())
            for l in layers:
                if l in self.rglru:
                    self.rglru[l] = torch.zeros_like(self.rglru[l])

        if store_type in ("conv", "all"):
            layers = [layer_idx] if layer_idx is not None else list(self.conv.keys())
            for l in layers:
                if l in self.conv:
                    self.conv[l] = torch.zeros_like(self.conv[l])

        if store_type in ("kv", "all"):
            layers = [layer_idx] if layer_idx is not None else list(self.kv.keys())
            for l in layers:
                if l in self.kv:
                    for k in self.kv[l]:
                        self.kv[l][k] = torch.zeros_like(self.kv[l][k])

    def to_cpu_dict(self) -> Dict[str, Any]:
        """Convert snapshot tensors to a detached CPU dictionary for serialization."""
        return {
            "rglru": {k: v.detach().cpu() for k, v in self.rglru.items()},
            "conv": {k: v.detach().cpu() for k, v in self.conv.items()},
            "kv": {k: {kk: vv.detach().cpu() for kk, vv in v.items()} for k, v in self.kv.items()},
            "cache_position": self.cache_position,
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_cpu_dict(
        cls,
        data: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "RecurrentStateSnapshot":
        """Reconstruct snapshot from CPU dictionary onto target device."""
        target_device = torch.device(device) if device is not None else torch.device("cpu")

        rglru = {int(k): v.to(target_device) for k, v in data["rglru"].items()}
        conv = {int(k): v.to(target_device) for k, v in data["conv"].items()}
        kv = {int(k): {kk: vv.to(target_device) for kk, vv in v.items()} for k, v in data["kv"].items()}

        dtype_str = data.get("dtype")
        parsed_dtype = getattr(torch, dtype_str.replace("torch.", "")) if dtype_str else None

        return cls(
            rglru=rglru,
            conv=conv,
            kv=kv,
            cache_position=int(data["cache_position"]),
            device=target_device,
            dtype=parsed_dtype,
            metadata=dict(data.get("metadata", {})),
        )

    def serialize(self) -> bytes:
        """Serialize snapshot to binary bytes."""
        buffer = io.BytesIO()
        torch.save(self.to_cpu_dict(), buffer)
        return buffer.getvalue()

    @classmethod
    def deserialize(
        cls,
        payload: bytes,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "RecurrentStateSnapshot":
        """Deserialize snapshot from binary bytes."""
        buffer = io.BytesIO(payload)
        data = torch.load(buffer, map_location="cpu", weights_only=True)
        return cls.from_cpu_dict(data, device=device)

    def distance(self, other: "RecurrentStateSnapshot") -> Dict[str, float]:
        """Compute layer-wise Frobenius and cosine distances against another snapshot."""
        dists: Dict[str, float] = {}

        # 1. RGLRU Distances
        rglru_fro = 0.0
        rglru_cos_sims = []
        for l in self.rglru:
            if l in other.rglru:
                diff = self.rglru[l].float() - other.rglru[l].float()
                rglru_fro += float(torch.norm(diff).item()) ** 2
                sim = F.cosine_similarity(
                    self.rglru[l].flatten().float(),
                    other.rglru[l].flatten().float(),
                    dim=0,
                ).item()
                rglru_cos_sims.append(sim)
        dists["rglru_frobenius"] = float(rglru_fro**0.5)
        dists["rglru_mean_cosine_sim"] = float(sum(rglru_cos_sims) / len(rglru_cos_sims)) if rglru_cos_sims else 1.0

        # 2. Conv Distances
        conv_fro = 0.0
        conv_cos_sims = []
        for l in self.conv:
            if l in other.conv:
                diff = self.conv[l].float() - other.conv[l].float()
                conv_fro += float(torch.norm(diff).item()) ** 2
                sim = F.cosine_similarity(
                    self.conv[l].flatten().float(),
                    other.conv[l].flatten().float(),
                    dim=0,
                ).item()
                conv_cos_sims.append(sim)
        dists["conv_frobenius"] = float(conv_fro**0.5)
        dists["conv_mean_cosine_sim"] = float(sum(conv_cos_sims) / len(conv_cos_sims)) if conv_cos_sims else 1.0

        # 3. KV Distances
        kv_fro = 0.0
        for l in self.kv:
            if l in other.kv:
                for k in ["key", "value"]:
                    if k in self.kv[l] and k in other.kv[l]:
                        diff = self.kv[l][k].float() - other.kv[l][k].float()
                        kv_fro += float(torch.norm(diff).item()) ** 2
        dists["kv_frobenius"] = float(kv_fro**0.5)

        return dists
