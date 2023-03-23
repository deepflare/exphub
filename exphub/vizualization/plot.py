from dataclasses import dataclass
from typing import Iterable, Optional, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from exphub.aggregators.aggregators import Vault
from exphub.aggregators.aggregators import AggregatorChain
import plotly.io as pio


@dataclass
class Grouping:
    df_hyperparams: pd.DataFrame
    col: str
    short_col_name: Optional[str] = None


@dataclass
class Series:
    df: pd.DataFrame
    subtitle: str
    yaxis_title: Optional[str] = None
    xaxis_title: str = 'Step'
    aggregators_title: str = 'Aggregators'
    _metric_name: Optional[str] = None
    smoothing: int = 1


class Plot:

    def __init__(self,
                 dfs_series: Union[Series, Iterable[Series]],
                 aggs: AggregatorChain = None,
                 groupby: Grouping = None,
                 meta_df: pd.DataFrame = None):
        if aggs is None:
            if groupby is not None:
                raise ValueError('If groupby is provided, aggs must also be provided.')

            if meta_df is None:
                raise ValueError('If one wants to perform per-run plot, meta_df must be provided.')

            groupby = Grouping(meta_df, 'sys/id', 'id')
            aggs = Vault.MEAN

        _dfs_series = [dfs_series] if isinstance(dfs_series, Series) else dfs_series

        for series in _dfs_series:
            series._metric = ''.join(series.df.columns[0].split('_')[:-1])

        self._subplots = [(df_.subtitle, df_.xaxis_title, df_.yaxis_title, df_.aggregators_title,
                           self._generate_go_figures(df_.df, aggs, groupby, df_.smoothing), df_._metric)
                          for df_ in _dfs_series]

        self._init_rendering()

    def _init_rendering(self):
        pio.renderers.default = "jpg"

    def _generate_go_figures(self, df_series, aggs: AggregatorChain, groupby: Grouping, smoothing: int):
        metric_name = ''.join(df_series.columns[0].split('_')[:-1])
        dfs = self._groupby(df_series, metric_name, groupby)
        dfs = list(map(lambda x: (aggs(x[0]), x[1]), dfs))

        go_figs = []

        for df, g in dfs:
            df = df.copy()
            if smoothing > 1:
                df = df.rolling(smoothing).mean()

            if g is not None:
                short_groupby_name = groupby.col.split(
                    '/')[-1] if groupby.short_col_name is None else groupby.short_col_name
                go_figs += list(
                    map(
                        lambda agg_name: go.Scatter(
                            x=df.index, y=df[agg_name], mode='lines', name=f'{short_groupby_name}__{g}::{agg_name}'),
                        aggs.labels()))
            else:
                go_figs += list(
                    map(lambda agg_name: go.Scatter(x=df.index, y=df[agg_name], mode='lines', name=f'{agg_name}'),
                        aggs.labels()))

        return go_figs

    def _groupby(self, df: pd.DataFrame, metric_name: str, groupby: Grouping = None):
        if groupby is None:
            return [(df, None)]

        gs = groupby.df_hyperparams.groupby(groupby.col).groups  # Values of the groupby column
        group2ids = {g: groupby.df_hyperparams.loc[gs[g]]['sys/id'].to_list() for g in gs}
        group2series_cols = {g: list(map(lambda x: f'{metric_name}_{x}', group2ids[g])) for g in gs}
        # Return df with only the columns that are in the group. On second position returns the group name
        res = []
        for g in gs:
            cols_not_in_current_group = set(df.columns) - set(group2series_cols[g])
            res.append((df.drop(cols_not_in_current_group, axis=1), g))
        assert len(res) > 0, f'No groups found for {groupby.col}'
        return res

    def _create_figure(self, subplot):
        fig = go.Figure()
        subtitle, xaxis, yaxis, legend_title, go_figs, _metric_name = subplot
        fig.add_traces(go_figs)
        fig.update_layout(
            title=subtitle,
            xaxis_title=xaxis,
            yaxis_title=yaxis if yaxis is not None else _metric_name,
            legend_title=legend_title)

    def render(self, title: Optional[str] = None):
        if len(self._subplots) == 1:
            fig = self._create_figure(self._subplots[0])
        else:
            fig = make_subplots(
                rows=1,
                cols=len(self._subplots),
                shared_xaxes=False,
                shared_yaxes=True,
                subplot_titles=[s[0] for s in self._subplots])
            grid = [(x, 1) for x in range(1, len(self._subplots) + 1)]

            for (x, y), (_, xaxis, yaxis, legend_title, go_figs, _metric_name) in zip(grid, self._subplots):
                for go_fig in go_figs:
                    fig.add_trace(go_fig, row=y, col=x)
                fig.update_yaxes(title_text=yaxis if yaxis is not None else _metric_name, row=y, col=x)
                fig.update_xaxes(title_text=xaxis, row=y, col=x)
            fig.update_layout(title=title, legend_title=legend_title)

        return fig
