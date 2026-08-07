"""bge-reranker-v2-m3 cross-encoder, loaded once per process and shared.

The reranker is what makes "retrieve 30, keep 8" safe. A bi-encoder scores the
query against a vector that was written without the query in hand; a
cross-encoder reads both together, so it can tell ``Stage 1`` advice from
``Stage 1+`` advice and can notice that a unit is about a different vehicle
entirely. Retrieving wide and cutting hard is only an improvement if something
between the two steps is actually better at judging relevance.

``BAAI/bge-reranker-v2-m3`` specifically, and not by default:

* **Multilingual is a requirement.** 8.5% of the corpus is French. An
  English-only reranker would systematically demote every French unit against
  an English query and vice-versa -- a recall failure that shows up as "the
  agent is worse for Quebec customers" and in no aggregate metric.
* **Local.** Cohere Rerank would ship customer-call text to a third party and
  add a network hop inside a synchronous, user-blocking call.
* **Small.** 568M params on MPS, not a 7B reranker on the DGX. The DGX is
  already contended by the 31B chat model; a second queue in front of a
  blocking call is where p99 latency goes to die.

Measured steady state, 30 pairs in one batch, MPS fp16, real retrieved text
(2026-08-06)::

    111 tok/pair  (kb_unit median, 336 chars)    424 ms
    180 tok/pair  (kb_unit p95,    591 chars)    685 ms
    254 tok/pair  (legacy page,    ~900 chars)  1024 ms

Cold start (weight load + first pass) is 7.6 s, so anything user-facing calls
:meth:`Reranker.warmup` before it accepts traffic. Latency tracks *tokens*, not
``max_length``: dropping ``max_length`` to 256 moved the 254-token case by
under 10% because the batch pads to its own longest member either way. The
knob that actually costs is the number of candidates.

Concurrency: one model instance, two locks. ``_load_lock`` makes the lazy load
idempotent under concurrent first calls. ``_infer_lock`` serialises forward
passes because a single shared module dispatching onto one MPS command queue
from several threads is not safe, and because two concurrent batches on the
same device are not faster than one batch of twice the size anyway.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from .. import config

# 512 tokens, not the model's 8192. The median knowledge unit is 336 chars and
# p95 is 591, so 512 tokens truncates essentially nothing while keeping the
# quadratic attention cost of a 30-pair batch bounded.
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))
# 32 so that the production shape -- one query against 30 candidates -- is a
# single forward pass. Measured on MPS fp16 over 30 short pairs: batch 8 =
# 163 ms, 16 = 141 ms, 30 = 122 ms, 32 = 123 ms. Splitting 30 pairs across two
# batches costs ~15% for no memory saving worth having.
RERANK_BATCH = int(os.environ.get("RERANK_BATCH", "32"))
# Documents longer than this are cut before tokenisation. Cheap guard against a
# pathological payload (a whole scraped page) blowing the batch up.
RERANK_MAX_CHARS = int(os.environ.get("RERANK_MAX_CHARS", "2000"))


@dataclass(frozen=True)
class RerankScore:
    """Where a document landed after reranking.

    ``index`` refers back into the list handed to :meth:`Reranker.rerank`, so
    callers keep their own objects instead of the reranker inventing a wrapper
    for them.
    """

    index: int
    score: float  # raw cross-encoder logit, roughly -11..+11
    probability: float  # sigmoid(score), in (0, 1) -- comparable across queries


def _sigmoid(x: float) -> float:
    # Branch on the sign so neither exp() overflows for large |x|.
    import math

    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class Reranker:
    """Lazy, thread-safe, batched cross-encoder scorer."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        *,
        max_length: int = RERANK_MAX_LENGTH,
        batch_size: int = RERANK_BATCH,
    ) -> None:
        self.model_name = model_name or config.RERANK_MODEL
        self._requested_device = device or config.RERANK_DEVICE
        self.max_length = max_length
        self.batch_size = batch_size
        self.device: str | None = None
        self.dtype: str | None = None
        self._tok = None
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # -- loading -------------------------------------------------------------
    def _resolve_device(self) -> str:
        if self._requested_device != "auto":
            return self._requested_device
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    import torch
                    from transformers import (
                        AutoModelForSequenceClassification,
                        AutoTokenizer,
                    )

                    device = self._resolve_device()
                    # fp16 on an accelerator, fp32 on CPU: CPU fp16 matmul falls
                    # back to a slow emulated path and is a net loss.
                    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
                    tok = AutoTokenizer.from_pretrained(self.model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name, dtype=dtype
                    )
                    model.eval()
                    model.to(device)
                    self._tok = tok
                    self._model = model
                    self.device = device
                    self.dtype = str(dtype).replace("torch.", "")
        return self._tok, self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> float:
        """Load weights and run one pass. Returns seconds spent.

        First-call latency includes weight load plus MPS kernel compilation and
        is several times steady state. Anything that reports a latency number,
        and any server that does not want to hand its first user that cost,
        calls this at startup.
        """
        t0 = time.perf_counter()
        self.score("warmup query", ["warmup document"])
        return time.perf_counter() - t0

    # -- scoring -------------------------------------------------------------
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Raw cross-encoder logits, one per document, in input order."""
        if not documents:
            return []
        tok, model = self._load()
        import torch

        docs = [(d or "")[:RERANK_MAX_CHARS] for d in documents]
        out: list[float] = []
        with self._infer_lock, torch.inference_mode():
            for i in range(0, len(docs), self.batch_size):
                chunk = docs[i : i + self.batch_size]
                batch = tok(
                    [query] * len(chunk),
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = model(**batch).logits.view(-1).float()
                out.extend(logits.cpu().tolist())
        return out

    def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[RerankScore]:
        """Score and sort. Ties break on original order, which keeps the
        upstream RRF ordering as the tiebreak rather than an arbitrary one."""
        scores = self.score(query, documents)
        ranked = [
            RerankScore(index=i, score=s, probability=_sigmoid(s))
            for i, s in enumerate(scores)
        ]
        ranked.sort(key=lambda r: (-r.score, r.index))
        return ranked[:top_k] if top_k else ranked


_default: Reranker | None = None
_default_lock = threading.Lock()


def get_reranker() -> Reranker:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Reranker()
    return _default


def rerank_enabled() -> bool:
    return config.RERANK_ENABLED


if __name__ == "__main__":  # self-check: correctness, latency, multilinguality
    import statistics

    rr = get_reranker()
    print(f"model={rr.model_name}")
    print(f"cold start (load + first pass): {rr.warmup():.2f}s "
          f"device={rr.device} dtype={rr.dtype}")

    q = "my UniCONNECT+ cable is not detected by the Windows flashing software"
    docs = [
        "Install the UniCONNECT+ drivers, then disable Windows Driver Signature "
        "Enforcement and reconnect the cable.",
        "Disabling Driver Signature Enforcement does not work while Secure Boot is "
        "enabled in the BIOS; Secure Boot must be turned off first.",
        "The Unitronic Classic Black T-Shirt is available in sizes S through XXL.",
        "Stage 1 software for the 2.0T EA888 Gen3 makes 300 hp on 91 octane.",
    ]
    for r in rr.rerank(q, docs):
        print(f"  logit={r.score:+7.3f} p={r.probability:.4f}  {docs[r.index][:70]}")
    assert rr.rerank(q, docs)[0].index in (0, 1), "cable docs must outrank a t-shirt"

    # Multilingual: the French paraphrase must beat an English near-miss.
    fr_q = "mon cable UniCONNECT+ n'est pas detecte par le logiciel de flash"
    fr_docs = [
        "Le cable UniCONNECT+ n'est pas reconnu: installez les pilotes puis "
        "desactivez la verification de signature des pilotes Windows.",
        "Unitronic sells performance intakes for the Audi S3.",
    ]
    fr = rr.rerank(fr_q, fr_docs)
    print(f"  FR  logit={fr[0].score:+7.3f} idx={fr[0].index}")
    assert fr[0].index == 0, "cross-lingual match failed"

    # Steady-state latency for the production shape: 30 pairs, one call. These
    # are short one-sentence documents; selfcheck.py measures the same shape
    # against real retrieved payloads, which are 2-3x longer and slower.
    pairs = (docs * 8)[:30]
    rr.score(q, pairs)  # discard the first timed run
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        rr.score(q, pairs)
        samples.append((time.perf_counter() - t0) * 1000)
    print(f"  30 pairs: median {statistics.median(samples):.0f} ms  "
          f"min {min(samples):.0f}  max {max(samples):.0f}  "
          f"(n=5, batch={rr.batch_size}, max_length={rr.max_length})")
    print("rerank.py self-check OK")
