from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List
import pandas as pd

@dataclass
class Experiment:
    df_meta: pd.DataFrame
    dfs_series: field(default_factory=dict) # metric_name -> df
    
    def filter_via_hyperparams(self, conditions: list) -> 'Experiment':
        df_meta = self.df_meta.copy()
        for fn in conditions:
            df_meta = df_meta[fn(df_meta)]
        
        # Filter series columns only if they are present in the meta df
        dfs_series = {}
        for metric_name, df in self.dfs_series.items():
            # Initialize new series from index
            new_series = pd.DataFrame(index=df.index)
            new_series.index.name = df.index.name
            
            for col in df.columns:
                # print(df_meta['sys/id'].values)
                if col.split('_')[-1] in df_meta['sys/id'].values:
                    new_series[col] = df[col].copy()
                    
            dfs_series[metric_name] = new_series
        
        return Experiment(df_meta, dfs_series)
    
    def split_by_columns(self, columns: List[str]) -> Dict[str, 'Experiment']:
        unique_values_by_columns = {col: self.df_meta[col].unique() for col in columns}
        splits = {}
        import copy
        for values in product(*unique_values_by_columns.values()):
            split_describtion = '\n'.join([f'{col} = {val}' for col, val in zip(columns, values)])
            experiment = copy.deepcopy(self)
            for col, val in zip(columns, values):
                experiment = experiment.filter_via_hyperparams([lambda df: df[col] == val])
            splits[split_describtion] = experiment
        return splits
