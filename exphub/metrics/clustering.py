from exphub.metrics.metric import Metric
from abc import abstractmethod
from typing import Any
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score


class ClusterExtrinsicMetric(Metric):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        pass


class ClusterIntrinsicMetric(Metric):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, embeddings, labels, **kwargs) -> Any:
        pass


class RandIndex(ClusterExtrinsicMetric):

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return rand_score(y_gt, y_pred)


class AdjustedRandIndex(ClusterExtrinsicMetric):

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return adjusted_rand_score(y_gt, y_pred)


class MutualInfoScore(ClusterExtrinsicMetric):

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return mutual_info_score(y_gt, y_pred)


class NormalizedMutualInfoScore(ClusterExtrinsicMetric):

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return normalized_mutual_info_score(y_gt, y_pred)


class AdjustedMutualInfoScore(ClusterExtrinsicMetric):

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return adjusted_mutual_info_score(y_gt, y_pred)


class FowlkesMallowsScore(ClusterExtrinsicMetric):

    def __call__(self, y_pred, y_gt, **kwargs) -> Any:
        return fowlkes_mallows_score(y_gt, y_pred)


class SilhouetteScore(ClusterIntrinsicMetric):

    def __call__(self, embeddings, labels, **kwargs) -> Any:
        if len(embeddings) != len(labels):
            raise ValueError(
                f"Embeddings and cluster y_gt must have the same length. Got {len(embeddings)} and {len(labels)}")
        return silhouette_score(embeddings, labels)


class CalinskiHarabaszScore(ClusterIntrinsicMetric):

    def __call__(self, embeddings, labels, **kwargs) -> Any:
        if len(embeddings) != len(labels):
            raise ValueError(
                f"Embeddings and cluster y_gt must have the same length. Got {len(embeddings)} and {len(labels)}")
        return calinski_harabasz_score(embeddings, labels)


class DaviesBouldinScore(ClusterIntrinsicMetric):

    def __call__(self, embeddings, labels, **kwargs) -> Any:
        if len(embeddings) != len(labels):
            raise ValueError(
                f"Embeddings and cluster y_gt must have the same length. Got {len(embeddings)} and {len(labels)}")
        return davies_bouldin_score(embeddings, labels)
