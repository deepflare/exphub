from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List
import pandas as pd


@dataclass
class Experiment:
    """
    A class representing an experiment with its parameters and series.

    Attributes:
        params (pd.DataFrame): A DataFrame containing the parameters of the experiment.
        series (Dict[str, pd.DataFrame]): A dictionary mapping metric names to DataFrames.
    """
    params: pd.DataFrame
    series: field(default_factory=dict)  # metric_name -> df

    def __str__(self) -> str:
        return f'Experiment Instance\n{len(self.params)} parameters: {self.params_names}\n including...\n\t * Attributes: {self.attributes_names}\n\t * Series: {self.series_names}'

    def subset_params(self, params_names_to_keep: List[str]) -> 'Experiment':
        """
        Returns a new Experiment instance with a subset of the parameters.

        Args:
            params_names_to_keep (List[str]): A list of the names of the parameters to keep.

        Returns:
            Experiment: A new Experiment instance with a subset of the parameters.
        """
        attributes_params = [p for p in params_names_to_keep if p in self.attributes_names]
        series_params = [p for p in params_names_to_keep if p in self.series_names]

        if any((p not in self.params_names and p not in self.attributes_names) for p in params_names_to_keep):
            raise ValueError(f'Invalid parameter name. Valid names are {self.params_names}')

        new_series = {s: self.series[s] for s in series_params}
        new_params = self.params[attributes_params + series_params]

        return Experiment(new_params, new_series)

    def subset_runs(self, runs_ids_to_keep: List[str]) -> 'Experiment':
        """
        Returns a new Experiment instance with a subset of the runs.

        Args:
            runs_ids_to_keep (List[str]): A list of the IDs of the runs to keep.

        Returns:
            Experiment: A new Experiment instance with a subset of the runs.
        """
        new_params = self.params[self.params[self.id_column_name].isin(runs_ids_to_keep)]
        new_series = {s: self.series[s].loc[runs_ids_to_keep] for s in self.series_names}

        return Experiment(new_params, new_series)

    def without(self, params_names_to_drop: List[str]) -> 'Experiment':
        all_params = self.params_names
        return self.subset([p for p in all_params if p not in params_names_to_drop])

    @property
    def series_names(self) -> List[str]:
        """
        Returns the names of the series.

        Returns:
            List[str]: A list of the names of the series.
        """
        return list(self.series.keys())

    @property
    def attributes_names(self) -> List[str]:
        """
        Returns the names of the attributes.

        Returns:
            List[str]: A list of the names of the attributes.
        """
        return [c for c in self.params.columns if c not in self.series_names]

    @property
    def params_names(self) -> List[str]:
        """
        Returns the names of the parameters.

        Returns:
            List[str]: A list of the names of the parameters.
        """
        return self.params.columns.tolist()

    @property
    def id_column_name(self) -> str:
        """
        Returns the name of the ID column.

        Returns:
            str: The name of the ID column.
        """
        return 'sys/id' if 'sys/id' in self.params.columns else 'id'

    def filter_via_hyperparams(self, conditions: list) -> 'Experiment':
        """
        Filters the experiment based on the given conditions.

        Args:
            conditions (list): A list of functions to filter the experiment's parameters.

        Returns:
            Experiment: A new Experiment instance with filtered parameters and series.
        """
        df_meta = self.params.copy()
        for fn in conditions:
            df_meta = df_meta[fn(df_meta)]

        # If no rows left, return empty experiment
        if len(df_meta) == 0:
            return Experiment(pd.DataFrame(), {})

        # Filter series columns only if they are present in the meta df
        series = {}

        # Shorten compatibility
        id_col = 'sys/id' if 'sys/id' in df_meta.columns else 'id'

        for metric_name, df in self.series.items():
            # Initialize new series from index
            new_series = pd.DataFrame(index=df.index)
            new_series.index.name = df.index.name

            for col in df.columns:
                if col.split('_')[-1] in df_meta[id_col].values:
                    new_series[col] = df[col].copy()

            series[metric_name] = new_series

        return Experiment(df_meta, series)

    def split_by_columns(self, columns: List[str]) -> Dict[str, 'Experiment']:
        """
        Splits the experiment into sub-experiments based on unique combinations of values in the specified columns.

        Args:
            columns (List[str]): A list of column names to split the experiment by.

        Returns:
            Dict[str, Experiment]: A dictionary mapping split descriptions to sub-experiments.
        """
        unique_values_by_columns = {col: self.params[col].unique() for col in columns}
        splits = {}
        import copy
        for values in product(*unique_values_by_columns.values()):
            split_describtion = '\n'.join([f'{col} = {val}' for col, val in zip(columns, values)])
            experiment = copy.deepcopy(self)
            for col, val in zip(columns, values):
                experiment = experiment.filter_via_hyperparams([lambda df: df[col] == val])
            splits[split_describtion] = experiment

        # Final filtering to remove empty experiments
        splits = {k: v for k, v in splits.items() if len(v.params) > 0}

        return splits
