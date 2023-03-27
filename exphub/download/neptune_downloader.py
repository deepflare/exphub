from exphub.download.downloader import Downloader
import pandas as pd
from typing import Optional, Union, List
import os
import neptune.new as neptune
from exphub.download.experiment import Experiment
from exphub.utils.noise import Suppressor


class NeptuneDownloader(Downloader):
    """
    A NeptuneDownloader class for downloading experiment data from Neptune.ai.

    Attributes:
        project_name (str): The name of the Neptune project.
        api_token (Optional[str]): The Neptune API token. If not provided, it should be set as an environment variable.
    """
    NEPTUNE_API_TOKEN = 'NEPTUNE_API_TOKEN'

    def __init__(self, project_name: str, api_token: Optional[str] = None):
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
                 columns: Optional[List[str]] = None,
                 recursive_param_names: Optional[List[str]] = None,
                 series: List[str] = []) -> Experiment:
        """
        Downloads experiment data from Neptune.ai based on specified filtering criteria.

        Args:
            id (Optional[Union[str, List[str]]]): The run ID(s) to filter by.
            state (Optional[Union[str, List[str]]]): The run state(s) to filter by.
            owner (Optional[Union[str, List[str]]]): The run owner(s) to filter by.
            tag (Optional[Union[str, List[str]]]): The run tag(s) to filter by.
            columns (Optional[List[str]]): A list of columns to include in the result.
            recursive_param_names (Optional[List[str]]): A list of recursive parameter names to include in the result.
            series (List[str]): A list of series to download.

        Returns:
            Experiment: An Experiment instance containing the downloaded data.
        """
        if all([id is None, state is None, owner is None, tag is None]):
            raise ValueError('At least one of id, state, owner, or tag must be provided.')
        df_meta = self.project.fetch_runs_table(owner=owner, id=id, state=state, tag=tag, columns=columns).to_pandas()
        dfs_series = {}
        for series_col in series:
            dfs_series[series_col] = self.download_series(series_col, id=id, state=state, owner=owner, tag=tag)

        self.recursive_param_names = recursive_param_names

        if recursive_param_names is not None:
            for param_name in recursive_param_names:
                param_short_name = param_name.split('/')[-1]
                param_id_value = self.params[param_name].to_list()

                df_meta_recursive = pd.DataFrame({
                    param_short_name: [
                        Suppressor.exec_no_stdout(self.project.fetch_runs_table, id=v) for v in param_id_value
                    ]
                })
                df_meta[param_short_name] = df_meta_recursive[param_short_name]

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
