from abc import ABC, abstractmethod
import pandas as pd

from exphub.download.experiment import Experiment


class Downloader(ABC):
    """
    An abstract base class for a data downloader.

    This class should be subclassed to create custom downloaders for specific data sources.
    """

    @abstractmethod
    def download(self, *args, **kwargs) -> Experiment:
        """
        Downloads data and returns it as an Experiment instance.

        This method should be implemented by subclasses to define the data downloading logic.

        Args:
            *args: Variable-length arguments specific to the downloader implementation.
            **kwargs: Arbitrary keyword arguments specific to the downloader implementation.

        Returns:
            Experiment: An Experiment instance containing the downloaded data.
        """
        pass
