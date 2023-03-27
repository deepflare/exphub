from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List
import pandas as pd


@dataclass
class Experiment:
    params: pd.DataFrame
    series: field(default_factory=dict)  # metric_name -> df

    def filter_via_hyperparams(self, conditions: list) -> 'Experiment':
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
