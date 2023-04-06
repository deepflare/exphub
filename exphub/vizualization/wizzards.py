from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union
from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from exphub.aggregators.aggregators import Vault
from exphub.aggregators.aggregators import AggregatorChain
import plotly.io as pio

from exphub.download.experiment import Experiment


@dataclass
class Grouping:
    """
    A class representing a grouping configuration for the SeriesWizard.

    Attributes:
        df_hyperparams (pd.DataFrame): A DataFrame containing the hyperparameters.
        col (str): The column name to group by.
        short_col_name (Optional[str]): The short column name for display purposes. Defaults to None.
    """
    df_hyperparams: pd.DataFrame
    col: str
    short_col_name: Optional[str] = None


@dataclass
class Series:
    """
    A class representing a series to be visualized in the SeriesWizard.

    Attributes:
        df (pd.DataFrame): A DataFrame containing the series data.
        subtitle (str): The subtitle for the series.
        yaxis_title (Optional[str]): The y-axis title. Defaults to None.
        xaxis_title (str): The x-axis title. Defaults to 'Step'.
        aggregators_title (str): The title for the aggregators. Defaults to 'Aggregators'.
        _metric_name (Optional[str]): The metric name. Defaults to None.
        smoothing (int): The smoothing factor for the series. Defaults to 1.
    """
    df: pd.DataFrame
    subtitle: str
    yaxis_title: Optional[str] = None
    xaxis_title: str = 'Step'
    aggregators_title: str = 'Aggregators'
    _metric_name: Optional[str] = None
    smoothing: int = 1


class Wizard(ABC):
    """
    An abstract base class for creating wizards to render various visualizations.
    """

    @abstractmethod
    def render(self, **kwargs):
        pass


class TableWizard(Wizard):

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def render(self, attributes_color: str = '#211b1b', series_color: str = '#022b11'):
        return self.experiment.params.style.set_properties(
            **{
                'background-color': attributes_color
            }, subset=self.experiment.attributes_names).set_properties(
                **{'background-color': series_color}, subset=self.experiment.series_names)


class SeriesWizard(Wizard):

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def render(self):
        if len(self.experiment.series_names) == 0:
            raise ValueError('No series to plot.')
        
        if len(self.experiment.series_names) > 1:
            return self._render_multiple_series()
        
        return self._render_single_series()

    def _render_single_series(self):
        fig = go.Figure()
        metric_name = self._infer_metric_name(self.experiment.series_names[0])
                
        xaxis_title = 'step'
        yaxis_title = metric_name
        title = f'{yaxis_title} per {xaxis_title}'
        
        # subtitle = 
        fig.add_traces(go_figs)
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=metric_name,
            legend_title='runs')
    
    def _render_multiple_series(self):
        pass
    
    def _infer_metric_name(self, series_name: str):
        return '_'.join(series_name.split('_')[:-1])
    

class SeriesWizardLegacy(Wizard):
    """
    A class used to generate plotly visualizations from a given set of data series.

    Methods
    -------
    set_default_renderer(renderer)
        Sets the default renderer for plotly visualizations.
    render(title=None)
        Renders the plotly visualization.
    """

    @classmethod
    def set_default_renderer(cls, renderer):
        """
        Sets the default renderer for plotly visualizations.

        Parameters
        ----------
        renderer : str
            The renderer to use for plotly visualizations.
        """
        pio.renderers.default = renderer

    def __init__(self,
                 series: Union[Series, Iterable[Series]],
                 aggs: AggregatorChain = None,
                 groupby: Grouping = None,
                 meta_df: pd.DataFrame = None):
        """
        Initializes a SeriesWizard instance.

        Parameters
        ----------
        series : Union[Series, Iterable[Series]]
            The series to plot.
        aggs : AggregatorChain, optional
            The aggregator chain to apply to the series data.
        groupby : Grouping, optional
            The grouping to apply for the plot.
        meta_df : pd.DataFrame, optional
            The DataFrame containing metadata for the series data.
        """
        if aggs is None:
            if groupby is not None:
                raise ValueError('If groupby is provided, aggs must also be provided.')

            if meta_df is None:
                raise ValueError('If one wants to perform per-run plot, meta_df must be provided.')

            groupby = Grouping(meta_df, 'sys/id', 'id') if 'sys/id' in meta_df.columns else Grouping(meta_df, 'id')
            aggs = Vault._NO_AGGR

        _series = [series] if isinstance(series, Series) else series

        for series in _series:
            series._metric = ''.join(series.df.columns[0].split('_')[:-1])

        self._subplots = [(df_.subtitle, df_.xaxis_title, df_.yaxis_title, df_.aggregators_title,
                           self._generate_go_figures(df_.df, aggs, groupby, df_.smoothing), df_._metric)
                          for df_ in _series]

    def _generate_go_figures(self, series, aggs: AggregatorChain, groupby: Grouping, smoothing: int):
        """
        Generates plotly go figures for the given series, aggregators, and grouping.

        Parameters
        ----------
        series : pd.DataFrame
            The series data to plot.
        aggs : AggregatorChain
            The aggregator chain to apply to the series data.
        groupby : Grouping
            The grouping to apply for the plot.
        smoothing : int
            The smoothing factor to apply to the series data.

        Returns
        -------
        list
            A list of plotly go figures.
        """
        metric_name = ''.join(series.columns[0].split('_')[:-1])
        dfs = self._groupby(series, metric_name, groupby)
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
        """
        Groups the given DataFrame by the specified Grouping.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to group.
        metric_name : str
            The metric name to use for the grouping.
        groupby : Grouping, optional
            The grouping to apply for the plot.

        Returns
        -------
        list
            A list of tuples containing the grouped DataFrame and group name.
        """
        if groupby is None:
            return [(df, None)]

        gs = groupby.df_hyperparams.groupby(groupby.col).groups  # Values of the groupby column
        id_col_name = 'sys/id' if 'sys/id' in df.columns else 'id'
        group2ids = {g: groupby.df_hyperparams.loc[gs[g]][id_col_name].to_list() for g in gs}
        group2series_cols = {g: list(map(lambda x: f'{metric_name}_{x}', group2ids[g])) for g in gs}
        # Return df with only the columns that are in the group. On second position returns the group name
        res = []
        for g in gs:
            cols_not_in_current_group = set(df.columns) - set(group2series_cols[g])
            res.append((df.drop(cols_not_in_current_group, axis=1), g))
        assert len(res) > 0, f'No groups found for {groupby.col}'
        return res

    def _create_figure(self, subplot):
        """
        Creates a plotly figure from the given subplot.

        Parameters
        ----------
        subplot : tuple
            The subplot information.

        Returns
        -------
        go.Figure
            A plotly figure.
        """
        fig = go.Figure()
        subtitle, xaxis, yaxis, legend_title, go_figs, _metric_name = subplot
        fig.add_traces(go_figs)
        fig.update_layout(
            title=subtitle,
            xaxis_title=xaxis,
            yaxis_title=yaxis if yaxis is not None else _metric_name,
            legend_title=legend_title)

    def render(self, title: Optional[str] = None):
        """
        Renders the plotly visualization.

        Parameters
        ----------
        title : Optional[str], optional
            The title for the visualization.

        Returns
        -------
        go.Figure
            A plotly figure.
        """
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
