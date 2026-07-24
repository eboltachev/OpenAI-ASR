from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class WeightedEmbedding:
    vector: FloatVector
    duration: float
    quality: float = 1.0


def l2_normalize(vector: FloatVector) -> FloatVector:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero speaker embedding")
    return (vector / norm).astype(np.float32)


def cosine_similarity(first: FloatVector, second: FloatVector) -> float:
    return float(l2_normalize(first) @ l2_normalize(second))


def aggregate_embeddings(
    samples: list[WeightedEmbedding], *, similarity_floor: float
) -> tuple[FloatVector, list[float]]:
    if not samples:
        raise ValueError("At least one speaker embedding is required")
    vectors = np.stack([l2_normalize(sample.vector) for sample in samples])
    initial = l2_normalize(vectors.mean(axis=0).astype(np.float32))
    similarities = vectors @ initial
    accepted = np.flatnonzero(similarities >= similarity_floor)
    if accepted.size == 0:
        accepted = np.array([int(np.argmax(similarities))])
    accepted_vectors = vectors[accepted]
    weights = np.asarray(
        [max(0.0, samples[index].quality) * max(0.0, samples[index].duration) for index in accepted],
        dtype=np.float32,
    )
    if float(weights.sum()) <= 1e-12:
        weights = np.ones_like(weights)
    centroid = np.average(accepted_vectors, axis=0, weights=weights).astype(np.float32)
    return l2_normalize(centroid), [float(value) for value in similarities]
