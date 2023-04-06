from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from exphub.download.experiment import Experiment


class Wizard(ABC):
    """
    An abstract base class for creating wizards to render various visualizations.
    """

    @abstractmethod
    def render(self, **kwargs):
        """
        Renders the visualization. This method should be implemented by subclasses.

        :param kwargs: Optional keyword arguments specific to each subclass.
        """
        pass


class TableWizard(Wizard):
    """
    A wizard class for rendering tables with custom styling. This class is used for visualizing
    experiment data by applying different background colors to attributes and series in the table.
    """

    def __init__(self, experiment: Experiment):
        """
        Initializes the TableWizard with the given experiment.

        :param experiment: The experiment object containing data to be visualized.
        """
        self.experiment = experiment

    def render(self, attributes_color: str = '#211b1b', series_color: str = '#022b11'):
        """
        Renders the table with the specified background colors for attributes and series.

        :param attributes_color: The background color to be applied to attribute names. Default is '#211b1b'.
        :param series_color: The background color to be applied to series names. Default is '#022b11'.
        :return: The styled table with the specified background colors applied.
        """
        return self.experiment.params.style.set_properties(
            **{
                'background-color': attributes_color
            }, subset=self.experiment.attributes_names).set_properties(
                **{'background-color': series_color}, subset=self.experiment.series_names)


class SeriesWizard(Wizard):
    """
    A wizard class for rendering line plots of time series data from experiments. This class is used
    for visualizing experiment data by plotting single or multiple series on a line chart or subplots.
    """

    def __init__(self, experiment: Experiment):
        """
        Initializes the SeriesWizard with the given experiment.

        :param experiment: The experiment object containing data to be visualized.
        """
        self.experiment = experiment

    def render(self):
        """
        Renders the line plot visualization for single or multiple series.

        :return: A plotly figure containing the line plot(s) of the time series data.
        """
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
