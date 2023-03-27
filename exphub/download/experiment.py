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
        for metric_name, df in self.series.items():
            # Initialize new series from index
            new_series = pd.DataFrame(index=df.index)
            new_series.index.name = df.index.name

            for col in df.columns:
                if col.split('_')[-1] in df_meta['sys/id'].values:
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
