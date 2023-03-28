from exphub.download.downloader import Downloader
import pandas as pd
from typing import Optional, Union, List
import os
import neptune.new as neptune
from exphub.download.experiment import Experiment
from exphub.utils.noise import Suppressor
from exphub.utils.paths import shorten_paths

class NeptuneDownloader(Downloader):
    """
    A NeptuneDownloader class for downloading experiment data from Neptune.ai.

    Attributes:
        project_name (str): The name of the Neptune project.
        api_token (Optional[str]): The Neptune API token. If not provided, it should be set as an environment variable.
    """
    NEPTUNE_API_TOKEN = 'NEPTUNE_API_TOKEN'

    def __init__(self, project_name: str, api_token: Optional[str] = None):
        """
        Download experiment data from Neptune.ai with the specified filters and settings.

        Args:
            id (Optional[Union[str, List[str]]]): A list of experiment IDs to download.
            state (Optional[Union[str, List[str]]]): A list of experiment states to download.
            owner (Optional[Union[str, List[str]]]): A list of experiment owners to download.
            tag (Optional[Union[str, List[str]]]): A list of experiment tags to download.
            attributes (Optional[List[str]]): A list of experiment attributes to download.
            short_names (bool): Whether to shorten the column names in the resulting data. Defaults to True.
            series (List[str]): A list of experiment series to download.

        Returns:
            Experiment: An Experiment object containing the downloaded data.
        """
        self.api_token = api_token
        self.project_name = project_name
        if self.api_token is None:
            if NeptuneDownloader.NEPTUNE_API_TOKEN not in os.environ:
                raise ValueError(f'Environment variable {NeptuneDownloader.NEPTUNE_API_TOKEN} not found.')
            self.api_token = os.environ[NeptuneDownloader.NEPTUNE_API_TOKEN]
        else:
            self.api_token = api_token
            os.environ[NeptuneDownloader.NEPTUNE_API_TOKEN] = api_token
        self.project = neptune.init_project(project=self.project_name, mode="read-only", api_token=self.api_token)

    def download(self,
                 id: Optional[Union[str, List[str]]] = None,
                 state: Optional[Union[str, List[str]]] = None,
                 owner: Optional[Union[str, List[str]]] = None,
                 tag: Optional[Union[str, List[str]]] = None,
                 attributes: Optional[List[str]] = None,
                 short_names: bool = True,
                 series: List[str] = []) -> Experiment:
        if all([id is None, state is None, owner is None, tag is None]):
            raise ValueError('At least one of id, state, owner, or tag must be provided.')
        columns = [*attributes, *series]
        df_meta = self.project.fetch_runs_table(owner=owner, id=id, state=state, tag=tag, columns=columns).to_pandas()
        dfs_series = {}
        for series_col in series:
            dfs_series[series_col] = self.download_series(series_col, id=id, state=state, owner=owner, tag=tag)

        self.short_names = short_names

        if short_names:
            # Modify df_meta in place
            meta_long2short = shorten_paths(df_meta)
            df_meta.rename(columns=meta_long2short, inplace=True)
            
            # Modify dfs_series in place
            short_df_series = {}
            for series_col, df in dfs_series.items():
                long2short = shorten_paths(df)
                short_df_series[meta_long2short[series_col]] = df.rename(columns=long2short)
            dfs_series = short_df_series
            
        return Experiment(df_meta, dfs_series)

    def download_series(self,
                        series_column: Union[List[str], str],
                        id: Optional[Union[str, List[str]]] = None,
                        state: Optional[Union[str, List[str]]] = None,
                        owner: Optional[Union[str, List[str]]] = None,
                        tag: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Downloads a specified series of data from Neptune.ai based on filtering criteria.

        Args:
            series_column (Union[List[str], str]): The name of the series to download.
            id (Optional[Union[str, List[str]]]): The run ID(s) to filter by.
            state (Optional[Union[str, List[str]]]): The run state(s) to filter by.
            owner (Optional[Union[str, List[str]]]): The run owner(s) to filter by.
            tag (Optional[Union[str, List[str]]]): The run tag(s) to filter by.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the downloaded series data.
        """
        if all([id is None, state is None, owner is None, tag is None]):
            raise ValueError('At least one of id, state, owner, or tag must be provided.')

        ids = self.project.fetch_runs_table(
            owner=owner, id=id, state=state, tag=tag, columns='sys/id').to_pandas()['sys/id'].values

        # Run initialization
        runs = [
            Suppressor.exec_no_stdout(
                neptune.init_run, project=self.project_name, with_id=run_id, mode="read-only", api_token=self.api_token)
            for run_id in ids
        ]

        def _fetch_values(col_label):
            if isinstance(col_label, list):
                assert len(col_label) == 1
                col_label = col_label[0]

            # Fetching values and counting the number of values
            id2value = {}
            missing = 0
            for id, run in zip(ids, runs):
                try:
                    id2value[id] = Suppressor.exec_no_stdout(run[col_label].fetch_values, include_timestamp=False)
                except neptune.exceptions.NeptuneException:
                    print(f'[WARNING] Run {id} does not have a column named {col_label}')
                    missing += 1
            if missing == len(ids):
                raise ValueError(f'No runs have a column named {col_label}')

            df = pd.DataFrame({})
            for id, value in id2value.items():
                df[f'{col_label}_{id}'] = value['value']

            return df

        if isinstance(series_column, str) or len(series_column) == 1:
            return _fetch_values(series_column)
        else:
            assert isinstance(series_column, list)
            dfs = [_fetch_values(col_label) for col_label in series_column]
            df = dfs[0]
            for d in dfs[1:]:
                df = df.join(d)
