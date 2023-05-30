from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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

    def __init__(self, experiment: Experiment, horizontal: bool = True, show_params: bool = False):
        """
        Initializes the SeriesWizard with the given experiment.

        :param experiment: The experiment object containing data to be visualized.
        """
        self.experiment = experiment
        self.horizontal = horizontal
        self.show_params = show_params

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
        to_merge = []
        for name, s in self.experiment.series.items():
            s_c = s.copy()
            s_c['metric'] = name
            s_c['step'] = s_c.index
            to_merge.append(s_c)
        df = pd.concat(to_merge)
        dff = df.melt(
            value_vars=df.columns.drop(['step', 'metric']),
            id_vars=['step', 'metric'],
            value_name='value',
            var_name='model')
        if self.horizontal:
            fig = px.line(dff, x='step', y='value', facet_col='metric', color='model')
        else:
            fig = px.line(dff, x='step', y='value', facet_row='metric', color='model')

        if self.show_params:
            params = self.experiment.params[self.experiment.attributes_names]
            for i, _ in enumerate(fig.data):
                row = params[params['id'] == fig.data[i]['name']].iloc[0]
                new_name = fig.data[i]['name'] + ': '
                for key, val in row.items():
                    if key != 'id':
                        new_name += f'{key}={val}; '
                fig.data[i]['name'] = new_name

        return fig
