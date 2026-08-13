"""Ollama Local LLM Backend with greedy deterministic options, SHA256 digest extraction, and retries."""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class OllamaBackend:
    """Interface for local LLM inference via Ollama REST API with fail-fast digest verification."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
        timeout: float = 120.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self._digest: Optional[str] = None
        self._verify_and_cache_model()

    def _verify_and_cache_model(self) -> None:
        """Strict verification against Ollama tags. Fails fast if model not present."""
        tags_url = f"{self.base_url}/api/tags"
        try:
            req = urllib.request.Request(tags_url, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise ConnectionError(
                f"Failed to reach Ollama daemon at {self.base_url}. "
                f"Ensure 'ollama serve' is active. Error: {e}"
            )

        models = data.get("models", [])
        matched = None
        for m in models:
            name = m.get("name", "")
            model_tag = m.get("model", "")
            if self.model_name in [name, model_tag, name.split(":")[0]]:
                matched = m
                break

        if not matched:
            raise RuntimeError(
                f"Requested model '{self.model_name}' is not installed in Ollama. "
                f"Available models: {[m.get('name') for m in models]}"
            )

        self._digest = matched.get("digest", "unknown")
        if self._digest == "unknown":
            raise RuntimeError(f"Could not retrieve deterministic SHA256 digest for '{self.model_name}'.")

    def get_digest(self) -> str:
        """Return the verified SHA256 model digest."""
        if not self._digest:
            self._verify_and_cache_model()
        return self._digest or "unknown"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Send chat messages to Ollama `/api/chat` with greedy deterministic options and retry."""
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

        last_err = None
        result = {}
        elapsed_ms = 0.0
        for attempt in range(3):
            try:
                start_time = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                elapsed_ms = (time.time() - start_time) * 1000.0
                break
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        else:
            raise TimeoutError(f"Ollama chat request failed after 3 attempts: {last_err}")

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

    def step(self, prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        """Single-turn prompt execution."""
        messages = [{"role": "user", "content": prompt}]
        content, metadata = self.chat(messages)
        state_hash = self.get_digest()[:16]
        return content, state_hash, metadata
