from loguru import logger
import numpy as np
from exphub.download.downloader import Downloader
import pandas as pd
from typing import Optional, Union, List, Tuple
import os
from neptune import Project, Run

from exphub.download.experiment import Experiment
from exphub.utils.logging import StdoutToLoguru
from exphub.utils.paths import shorten_paths

from neptune.exceptions import TypeDoesNotSupportAttributeException, MissingFieldException


class NeptuneDownloader(Downloader):
    """
    A NeptuneDownloader class for downloading experiment data from Neptune.ai.

    Attributes:
        project_name (str): The name of the Neptune project.
        api_token (Optional[str]): The Neptune API token. If not provided, it should be set as an environment variable.
    """
    NEPTUNE_API_TOKEN = 'NEPTUNE_API_TOKEN'
    EXPHUB_CACHE = '.exphub_cache'

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
        
        self.project = Project(project=self.project_name, api_token=self.api_token, mode="read-only")
        
        # Cache setup
        if not os.path.exists(NeptuneDownloader.EXPHUB_CACHE):
            os.mkdir(NeptuneDownloader.EXPHUB_CACHE)

    def _filter_ids_and_params(self, 
                            ids: List[str], 
                            params: pd.DataFrame,
                            required_columns: List[str]
                            ) -> Tuple[List[str], pd.DataFrame]:
        """
        Filters out ids and params of runs that do not have all required_columns.
        """
        with StdoutToLoguru():
            runs = [
                Run(run_id, project=self.project_name, api_token=self.api_token, mode="read-only")
                for run_id in ids
            ]

        ret_ids = []
        for id_, run in zip(ids, runs):
            remove = False
            for col_label in required_columns:
                try:
                    # I don't know of a better way to check if a column exists
                    run[col_label].fetch_last()
                except (KeyError, TypeDoesNotSupportAttributeException, MissingFieldException):
                    remove = True
                    break

            if not remove:
                ret_ids.append(id_)

        params = params[params['sys/id'].isin(ret_ids)]

        return ret_ids, params

    def download(self,
                 id: Optional[Union[str, List[str]]] = None,
                 state: Optional[Union[str, List[str]]] = None,
                 owner: Optional[Union[str, List[str]]] = None,
                 tag: Optional[Union[str, List[str]]] = None,
                 attributes: Optional[List[str]] = None,
                 short_names: bool = True,
                 series: List[str] = [],
                 required_columns: Optional[List[str]] = None,) -> Experiment:
        if all([id is None, state is None, owner is None, tag is None]):
            raise ValueError('At least one of id, state, owner, or tag must be provided.')
        
        if attributes is None:
            columns = None
        else:
            columns = [*attributes, *series]
        
        params = self.project.fetch_runs_table(owner=owner, id=id, state=state, tag=tag, columns=columns).to_pandas()
        assert len(params) > 0, "No experiments found"
        ids = params['sys/id'].values

        if required_columns is not None:
            ids, params = self._filter_ids_and_params(ids, params, required_columns)

        series_dict = {}
        for series_col in series:
            series_dict[series_col] = self._download_series(series_col, ids)

        self.short_names = short_names

        if short_names:
            # Modify params in place
            meta_long2short = shorten_paths(params)
            params.rename(columns=meta_long2short, inplace=True)

            # Modify series in place
            short_df_series = {}
            for series_col, df in series_dict.items():
                long2short = shorten_paths(df)
                short_df_series[meta_long2short[series_col]] = df.rename(columns=long2short)
            series_dict = short_df_series

        # Transformning ids dtype from objects to strings
        exp = Experiment(params, series_dict)
        exp.params[exp.id_column_name] = exp.params[exp.id_column_name].astype(str)
        logger.info(f'dtypes of params: {exp.params.dtypes}')
        
        return exp

    def _download_series(self,
                         series_column: Union[List[str], str],
                         ids: List[str]) -> pd.DataFrame:
        """
        Downloads a specified series of data from Neptune.ai based on filtering criteria.

        Args:
            series_column (Union[List[str], str]): The name of the series to download.
            id (List[str]): The run IDs to download.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the downloaded series data.
        """
        # Run initialization
        with StdoutToLoguru():
            runs = [
                Run(run_id, project=self.project_name, api_token=self.api_token, mode="read-only")
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
                    id2value[id] = run[col_label].fetch_values(include_timestamp=False)
                except (KeyError, TypeDoesNotSupportAttributeException):
                    print(f'[WARNING] Run {id} does not have a column named {col_label}')
                    missing += 1
            if missing == len(ids):
                raise ValueError(f'No runs have a column named {col_label}')

            # Pad all series to the max length
            max_len = max([len(value['value']) for value in id2value.values()])

            df_dict = {id : np.pad(value['value'], (0, max_len - len(value['value'])), 'constant', constant_values=np.nan) for id, value in id2value.items()}
            df = pd.DataFrame(df_dict)
        
            return df

        return _fetch_values(series_column)


class NeptuneDownloaderParentExpLinked(NeptuneDownloader):

    def __init__(self, project_name: str, api_token: Optional[str] = None, param_name_of_parent_exp_id: str = None):
        super().__init__(project_name, api_token)
        self.param_name_of_parent_exp_id = param_name_of_parent_exp_id

    def download(self,
                 id: Optional[Union[str, List[str]]] = None,
                 state: Optional[Union[str, List[str]]] = None,
                 owner: Optional[Union[str, List[str]]] = None,
                 tag: Optional[Union[str, List[str]]] = None,
                 attributes: Optional[List[str]] = None,
                 short_names: bool = True,
                 series: List[str] = [],
                 attributes_parent: List[str] = [],
                 series_parent: List[str] = []) -> Experiment:
        """
        Download the experiment data linked to a parent experiment in Neptune and merge the parameters and
        series from both experiments.

        Args:
            id (Optional[Union[str, List[str]]]): Experiment ID(s) to download (default: None).
            state (Optional[Union[str, List[str]]]): Experiment state(s) to download (default: None).
            owner (Optional[Union[str, List[str]]]): Experiment owner(s) to download (default: None).
            tag (Optional[Union[str, List[str]]]): Experiment tag(s) to download (default: None).
            attributes (Optional[List[str]]): Experiment attributes to download (default: None).
            short_names (bool): Whether to use short names for experiment attributes (default: True).
            series (List[str]): Series to download for the current experiment (default: []).
            attributes_parent (List[str]): Parent experiment attributes to download (default: []).
            series_parent (List[str]): Series to download for the parent experiment (default: []).

        Returns:
            Experiment: An Experiment object containing the combined parameters and series from the current
                        and parent experiments.
        """
        current_experiment = super().download(id, state, owner, tag, attributes, short_names, series)
        parent_experiment = self._download_parent_experiment(current_experiment, attributes_parent, series_parent)

        # Collapse params and metrics of parent expeirment into current_experiment, 'parent' namespace
        parent_experiment.params = parent_experiment.params.add_prefix('parent/')
        parent_experiment.series = {k: v.add_prefix('parent/') for k, v in parent_experiment.series.items()}

        print(f'parent experiment cols: {parent_experiment.params.columns}')
        print(f'current experiment cols: {current_experiment.params.columns}')

        print(f'ids of parent experiment: {parent_experiment.params["parent/id"].values.tolist()}')
        print(f'ids of current experiment: {current_experiment.params["id"].values.tolist()}')

        # Merge params of current experiment and parent experiment by equality of id and parent/id
        new_params = current_experiment.params.merge(
            parent_experiment.params, left_on=[self.param_name_of_parent_exp_id], right_on=['parent/id'])

        # TODO: support merging series
        return Experiment(new_params, current_experiment.series)

    def _download_parent_experiment(self, current_experiment, attributes_parent: List[str], series_parent: List[str]):
        """
        Download the parent experiment data using the parent experiment ID from the current experiment.

        Args:
            current_experiment (Experiment): The current experiment.
            attributes_parent (List[str]): Parent experiment attributes to download.
            series_parent (List[str]): Series to download for the parent experiment.

        Returns:
            Experiment: The parent experiment data.
        """
        parent_experiment_id = current_experiment.params[self.param_name_of_parent_exp_id].values.tolist()
        print(f'Downloading parent experiment {parent_experiment_id}...')
        return super().download(id=parent_experiment_id, attributes=attributes_parent, series=series_parent)
