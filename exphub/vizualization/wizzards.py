from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        metric_name = self.experiment.series_names[0]

        xaxis_title = 'step'
        yaxis_title = metric_name
        title = f'{yaxis_title} per {xaxis_title}'

        traces = self._generate_traces(self.experiment.series[self.experiment.series_names[0]])

        fig.add_traces(traces)
        fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=metric_name, legend_title='runs')

        return fig

    def _generate_traces(self, series: pd.DataFrame):
        traces = []
        for run_id in series.columns:
            traces.append(go.Scatter(x=series.index, y=series[run_id], name=run_id))
        return traces

    def _render_multiple_series(self):
        n_plots = len(self.experiment.series_names)
        xaxis_title = 'step'

        fig = make_subplots(
            rows=1,
            cols=n_plots,
            shared_xaxes=False,
            shared_yaxes=True,
            subplot_titles=[f'{metric_name} per step' for metric_name in self.experiment.series_names])
        grid = [(x, 1) for x in range(1, n_plots + 1)]

        for (x, y), series_name in zip(grid, self.experiment.series_names):
            traces = self._generate_traces(self.experiment.series[series_name])
            fig.add_traces(traces, rows=y, cols=x)
            fig.update_yaxes(title_text=series_name, row=y, col=x)
            fig.update_xaxes(title_text=xaxis_title, row=y, col=x)

        return fig
