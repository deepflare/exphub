from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        return super().__call__(**kwargs)


class MetricsCollection:

    def __init__(self, metrics: list[Metric]) -> None:
        super().__init__()
        self.metrics = metrics

    def __call__(self, **kwargs) -> Any:
        return {metric.__class__.__name__: metric(**kwargs) for metric in self.metrics}
