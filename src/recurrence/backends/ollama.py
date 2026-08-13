"""Ollama REST API backend for reproducible open model scouting."""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple


class OllamaBackend:
    """Interface to local Ollama API for deterministic open model scouting."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
        timeout: int = 60,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.model_info = self._fetch_model_info()
        if self.model_info["digest"] == "unknown":
            raise RuntimeError(
                f"Failed to verify model '{self.model_name}' on Ollama at {self.base_url}. "
                "Ensure Ollama is running and the model has been pulled (`ollama pull {self.model_name}`)."
            )

    def _fetch_model_info(self) -> Dict[str, Any]:
        """Fetch model metadata, digest, and details from Ollama API with strict verification."""
        info: Dict[str, Any] = {"digest": "unknown", "details": {}, "template": ""}
        
        # 1. Fetch digest from /api/tags
        try:
            tags_req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(tags_req, timeout=5) as response:
                tags_data = json.loads(response.read().decode("utf-8"))
                for m in tags_data.get("models", []):
                    if m.get("name") == self.model_name or m.get("model") == self.model_name:
                        info["digest"] = m.get("digest", "unknown")
                        info["details"] = m.get("details", {})
                        break
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama server at {self.base_url}/api/tags: {e}"
            ) from e

        # 2. Fetch template from /api/show
        try:
            show_payload = json.dumps({"name": self.model_name}).encode("utf-8")
            show_req = urllib.request.Request(
                f"{self.base_url}/api/show", data=show_payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(show_req, timeout=5) as response:
                show_data = json.loads(response.read().decode("utf-8"))
                info["template"] = show_data.get("template", "")
        except Exception:
            pass

        return info

    def get_digest(self) -> str:
        """Return verified model SHA256 digest string."""
        return self.model_info["digest"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Send chat completion request to Ollama /api/chat.

        Args:
            messages: List of message dicts [{'role': 'user', 'content': '...'}]
            temperature: Override sampling temperature
            seed: Override deterministic seed

        Returns:
            Tuple of (assistant_response_text, execution_metadata_dict)
        """
        temp = temperature if temperature is not None else self.temperature
        sd = seed if seed is not None else self.seed

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "seed": sd,
                "top_p": 1.0 if temp == 0.0 else 0.9,
            },
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data_bytes, headers={"Content-Type": "application/json"}
        )

        start_time = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        elapsed_ms = (time.time() - start_time) * 1000.0

        content = result.get("message", {}).get("content", "")
        metadata = {
            "model": self.model_name,
            "digest": self.get_digest(),
            "temperature": temp,
            "seed": sd,
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "eval_count": result.get("eval_count", 0),
            "total_duration_ms": elapsed_ms,
        }

        return content, metadata
