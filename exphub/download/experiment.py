from collections import defaultdict
from dataclasses import dataclass, field
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
                if col.split('_')[-1] in df_meta['sys/id'].values:
                    new_series[col] = df[col]
                    
            dfs_series[metric_name] = new_series
        
        return Experiment(df_meta, dfs_series)