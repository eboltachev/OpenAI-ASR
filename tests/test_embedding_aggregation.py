import numpy as np

from app.domain.embeddings import WeightedEmbedding, aggregate_embeddings, l2_normalize


def test_duration_weighted_centroid_is_normalized() -> None:
    first = np.asarray([1.0, 0.0], dtype=np.float32)
    second = np.asarray([0.8, 0.2], dtype=np.float32)
    centroid, similarities = aggregate_embeddings(
        [WeightedEmbedding(first, 3.0), WeightedEmbedding(second, 1.0)],
        similarity_floor=0.0,
    )
    assert np.isclose(np.linalg.norm(centroid), 1.0)
    assert len(similarities) == 2
    assert centroid[0] > centroid[1]
    assert np.allclose(l2_normalize(centroid), centroid)
