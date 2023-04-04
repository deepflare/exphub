from exphub.metrics.metric import Metric
from abc import abstractmethod
from typing import Any
from sklearn.metrics import (rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, fowlkes_mallows_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)


class ClusterExtrinsicMetric(Metric):
    """
    An abstract base class for clustering extrinsic metrics. Extrinsic metrics evaluate the
    quality of a clustering algorithm by comparing its output to a ground truth clustering.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        pass


class ClusterIntrinsicMetric(Metric):
    """
    An abstract base class for clustering intrinsic metrics. Intrinsic metrics evaluate the
    quality of a clustering algorithm without reference to a ground truth clustering.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, embeddings, labels, **kwargs) -> Any:
        pass


class RandIndex(ClusterExtrinsicMetric):
    """
    Calculates the Rand Index, a measure of the similarity between two clusterings.
    """

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return rand_score(y_gt, y_pred)


class AdjustedRandIndex(ClusterExtrinsicMetric):
    """
    Calculates the Adjusted Rand Index, a measure of the similarity between two clusterings
    that is adjusted for chance.
    """

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return adjusted_rand_score(y_gt, y_pred)


class MutualInfoScore(ClusterExtrinsicMetric):
    """
    Calculates the Mutual Information Score, a measure of the similarity between two clusterings
    based on the amount of shared information.
    """

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return mutual_info_score(y_gt, y_pred)


class NormalizedMutualInfoScore(ClusterExtrinsicMetric):
    """
    Calculates the Normalized Mutual Information Score, a measure of the similarity between
    two clusterings based on the normalized amount of shared information.
    """

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return normalized_mutual_info_score(y_gt, y_pred)


class AdjustedMutualInfoScore(ClusterExtrinsicMetric):
    """
    Calculates the Adjusted Mutual Information Score, a measure of the similarity between
    two clusterings based on the amount of shared information, adjusted for chance.
    """

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return adjusted_mutual_info_score(y_gt, y_pred)


class FowlkesMallowsScore(ClusterExtrinsicMetric):
    """
    Calculates the Fowlkes-Mallows Score, a measure of the similarity between two clusterings
    based on the geometric mean of precision and recall.
    """

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return fowlkes_mallows_score(y_gt, y_pred)


class SilhouetteScore(ClusterIntrinsicMetric):
    """
    Calculates the Silhouette Score, a measure of how similar an object is to its own cluster
    compared to other clusters. A higher score indicates better clustering quality.
    """

    def __call__(self, embeddings, labels, **kwargs) -> Any:
        if len(embeddings) != len(labels):
            raise ValueError(
                f"Embeddings and cluster y_gt must have the same length. Got {len(embeddings)} and {len(labels)}")
        return silhouette_score(embeddings, labels)


class CalinskiHarabaszScore(ClusterIntrinsicMetric):
    """
    Calculates the Calinski-Harabasz Score, also known as the Variance Ratio Criterion. This score
    is a measure of cluster dispersion, where a higher value indicates better clustering quality.
    The score is defined as the ratio of the between-cluster dispersion to the within-cluster dispersion.
    """

    def __call__(self, embeddings, labels, **kwargs) -> Any:
        if len(embeddings) != len(labels):
            raise ValueError(
                f"Embeddings and cluster y_gt must have the same length. Got {len(embeddings)} and {len(labels)}")
        return calinski_harabasz_score(embeddings, labels)


class DaviesBouldinScore(ClusterIntrinsicMetric):
    """
    Calculates the Davies-Bouldin Score, a measure of cluster quality based on the average
    similarity between clusters. A lower score indicates better clustering quality.
    The score is defined as the average of the maximum similarity between each cluster and
    all other clusters, where similarity is the ratio of within-cluster distances to between-cluster distances.
    """

    def __call__(self, embeddings, labels, **kwargs) -> Any:
        if len(embeddings) != len(labels):
            raise ValueError(
                f"Embeddings and cluster y_gt must have the same length. Got {len(embeddings)} and {len(labels)}")
        return davies_bouldin_score(embeddings, labels)
