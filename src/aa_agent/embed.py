"""Text embedding, behind a protocol.

The protocol exists so clustering and taxonomy logic can be tested with a
deterministic stub instead of downloading a 50MB ONNX model. CI must never
depend on a model download -- same principle as never depending on a live
free-tier API.

Embeddings are cached to disk as .npy keyed by (model, dtype, content hash).
Re-running intent discovery after a crash or a parameter tweak should not
re-embed 60k messages.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a list of strings into a (n, dim) float array."""

    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    """ONNX-backed embedder (~50MB installed).

    Imported lazily inside __init__ so that merely importing this module --
    which the test suite does -- never pulls in onnxruntime.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        dim: int = 384,
        parallel: int | None = 0,
    ) -> None:
        from fastembed import TextEmbedding  # noqa: PLC0415

        self.name = model
        self.dim = dim
        # parallel=0 -> use all cores; >1 -> that many; None -> single-core
        # onnxruntime threading (fastembed's default).
        #
        # Configurable rather than hard-coded because data-parallel encoding
        # spawns worker PROCESSES, and Windows uses spawn rather than fork:
        # every worker re-imports the module and loads its own ~67MB copy of
        # the ONNX model. For small batches that startup cost can exceed the
        # compute it saves, so `--parallel none` is a real escape hatch, not
        # a formality. Measure with a small --sample before a full run.
        self.parallel = parallel
        self._model = TextEmbedding(model_name=model)

    def encode(self, texts: list[str]) -> np.ndarray:
        # `parallel` is a per-call kwarg on .embed(), not on TextEmbedding()'s
        # constructor -- confirmed against the installed package's actual
        # signature, not just its docs, since docs have drifted from
        # behavior twice already in this project (DECISIONS.md #13, #20).
        #
        # `.embed()` returns a generator rather than a materialized list, so
        # wrapping it in a progress bar gives real incremental feedback
        # instead of one static spinner for the whole call -- which is the
        # entire reason "is this stuck or just slow?" was unanswerable.
        from rich.progress import (  # noqa: PLC0415
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
        )

        vectors: list[np.ndarray] = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("embedding", total=len(texts))
            for v in self._model.embed(texts, parallel=self.parallel):
                vectors.append(v)
                progress.update(task, advance=1)
        return np.asarray(vectors, dtype=np.float32)


def content_hash(texts: list[str], model: str, dtype: str) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{dtype}|{len(texts)}".encode())
    for t in texts:
        h.update(t.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed_cached(
    texts: list[str],
    embedder: Embedder,
    cache_dir: Path,
    dtype: str = "float16",
) -> np.ndarray:
    """Embed `texts`, reusing a cached array when the exact same input was
    embedded before with the same model and dtype.

    Stored as float16 by config default: for 60k x 384 that is ~46MB rather
    than ~92MB, and the precision loss is irrelevant for cosine similarity
    over short tweets. Returned as float32 because sklearn silently upcasts
    anyway and doing it once here is cheaper than per-call.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = content_hash(texts, embedder.name, dtype)
    path = cache_dir / f"emb_{key}.npy"

    if path.exists():
        try:
            return np.load(path).astype(np.float32)
        except (ValueError, EOFError, OSError):
            # A file left over from before the atomic-write fix (or any
            # other partial write) can't be trusted. Treat it as a miss and
            # re-embed rather than crash on a stale artifact the user has
            # no reason to know is bad.
            path.unlink(missing_ok=True)

    vectors = embedder.encode(texts)
    if vectors.shape[0] != len(texts):
        raise ValueError(f"embedder returned {vectors.shape[0]} vectors for {len(texts)} texts")

    # np.save is NOT atomic. A run killed mid-write (Ctrl+C, crash, OOM)
    # can leave a truncated .npy sitting at the real cache path -- which a
    # future run then tries to np.load() and fails on with a confusing
    # numpy error, nowhere near "the cache is corrupted" as an explanation.
    # Writing to a temp file in the same directory and renaming into place
    # means the real path only ever exists as a complete, valid file.
    # np.save silently appends ".npy" to any filename that doesn't already
    # end with it -- name the temp file so it already ends in .npy, or
    # numpy writes "*.npy.tmp.npy" and the rename below finds nothing at
    # the path it expects. Confirmed against the installed numpy directly
    # rather than assumed, after this exact thing broke it once already.
    tmp_path = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp_path, vectors.astype(dtype))
    tmp_path.replace(path)

    return vectors.astype(np.float32)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length so Euclidean k-means behaves like
    cosine clustering -- which is what we want for text embeddings, where
    magnitude carries no meaning."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms
