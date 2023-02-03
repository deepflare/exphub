from abc import ABC, abstractmethod
import pandas as pd


class Downloader(ABC):

    @abstractmethod
    def download(self, *args, **kwargs) -> pd.DataFrame:
        pass
