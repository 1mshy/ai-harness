"""Qwen3 embeddings with two interchangeable backends.

Qwen3 embedding models are asymmetric: the query side takes an instruct prefix,
the document side does not. Calling the document path for a query is a silent
recall loss -- everything still returns results, they are just worse. The two
entry points here are named ``embed_documents`` and ``embed_query`` so that a
call site cannot get it wrong by omission.

Backends, measured 2026-08-05:

    local  sentence-transformers on MPS      256 texts/s
    remote vLLM /v1/embeddings on the DGX    121 texts/s (contended by the 31B
                                             chat model on the same GPU)

Cross-backend cosine agreement is 1.0000, so an index built on one is queryable
from the other. Local wins on throughput because the DGX endpoint is sharing a
GPU with a 31B model at 0.65 memory utilisation.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from typing import Sequence

import numpy as np

from .. import config


class Embedder:
    """Lazy, thread-safe, backend-agnostic embedder."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or config.EMBED_BACKEND
        self._model = None
        self._lock = threading.Lock()

    # -- local ---------------------------------------------------------------
    def _local(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import SentenceTransformer

                    device = config.EMBED_DEVICE
                    if device == "auto":
                        if torch.backends.mps.is_available():
                            device = "mps"
                        elif torch.cuda.is_available():
                            device = "cuda"
                        else:
                            device = "cpu"
                    kwargs = {}
                    if device != "cpu":
                        kwargs["torch_dtype"] = torch.float16
                    self._model = SentenceTransformer(
                        config.EMBED_MODEL, device=device, model_kwargs=kwargs
                    )
        return self._model

    # -- remote --------------------------------------------------------------
    def _remote(self, texts: Sequence[str]) -> np.ndarray:
        out: list[list[float]] = []
        # vLLM batches internally but a single oversized request will time out
        # on a contended box; chunk so a slow batch cannot stall the build.
        for i in range(0, len(texts), 256):
            chunk = list(texts[i : i + 256])
            req = urllib.request.Request(
                f"{config.EMBED_REMOTE_URL}/embeddings",
                data=json.dumps(
                    {"model": config.EMBED_REMOTE_MODEL, "input": chunk}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.load(resp)
            out.extend(d["embedding"] for d in sorted(body["data"], key=lambda d: d["index"]))
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    # -- public --------------------------------------------------------------
    def _encode(self, texts: Sequence[str], prompt: str | None) -> np.ndarray:
        if not texts:
            return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
        if self.backend == "remote":
            # vLLM's OpenAI embedding route has no prompt/instruction argument,
            # so the instruction is prepended into the text itself. Qwen3 was
            # trained with the instruction inline, so this is equivalent to
            # what sentence-transformers does with prompt=.
            payload = [f"{prompt}{t}" for t in texts] if prompt else list(texts)
            return self._remote(payload)
        model = self._local()
        kwargs = {"prompt": prompt} if prompt else {}
        vecs = model.encode(
            list(texts),
            batch_size=config.EMBED_BATCH,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            **kwargs,
        )
        return vecs.astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Document side -- no instruct prefix."""
        return self._encode(texts, None)

    def embed_query(self, text: str) -> np.ndarray:
        """Query side -- instruct prefix applied."""
        return self._encode([text], config.QUERY_INSTRUCTION)[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, config.QUERY_INSTRUCTION)


_default: Embedder | None = None
_default_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Embedder()
    return _default
