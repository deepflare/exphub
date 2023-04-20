import pytest
import pandas as pd
import numpy as np
from exphub.download.experiment import Experiment

# Create a sample Experiment instance for testing
params_data = {
    'id': ['run1', 'run2', 'run3'],
    'attr1': [1, 2, 3],
    'attr2': [2, 3, 1],
    'metric1': [0.1, 0.2, 0.3],
    'metric2': [0.3, 0.2, 0.1],
}
params_df = pd.DataFrame(params_data)
sample_series = {
    'metric1': pd.DataFrame(data=[[0.1, 0.2, 0.3]], columns=['run1', 'run2', 'run3']),
    'metric2': pd.DataFrame(data=[[0.3, 0.2, 0.1]], columns=['run1', 'run2', 'run3']),
}
sample_experiment = Experiment(params=params_df, series=sample_series)


@pytest.mark.parametrize(
    "params_names_to_keep,expected",
    [
        (['attr1'], 1),
        (['attr2'], 1),
        (['metric1'], 1),
        (['attr1', 'attr2'], 2),
        (['attr1', 'metric1'], 2),
    ],
)
def test_subset_params(params_names_to_keep, expected):
    result = sample_experiment.subset_params(params_names_to_keep)
    assert len(result.params.columns) == expected


@pytest.mark.parametrize(
    "runs_ids_to_keep,expected",
    [
        (['run1'], 1),
        (['run2'], 1),
        (['run1', 'run2'], 2),
    ],
)
def test_subset_runs(runs_ids_to_keep, expected):
    result = sample_experiment.subset_runs(runs_ids_to_keep)
    assert len(result.series[sample_experiment.series_names[0]].columns) == expected


@pytest.mark.parametrize(
    "params_names_to_drop,expected",
    [
        (['attr1'], 4),
        (['attr2'], 4),
        (['metric1'], 4),
        (['attr1', 'attr2'], 3),
        (['attr1', 'metric1'], 3),
    ],
)
def test_without(params_names_to_drop, expected):
    result = sample_experiment.without(params_names_to_drop)
    assert len(result.params.columns) == expected


@pytest.mark.parametrize(
    "conditions,expected",
    [
        ([lambda df: df['attr1'] > 1], 2),
        ([lambda df: df['attr2'] > 1], 2),
        ([lambda df: df['attr1'] > 1, lambda df: df['attr2'] > 1], 1),
    ],
)
def test_filter_via_hyperparams(conditions, expected):
    result = sample_experiment.filter_via_hyperparams(conditions)
    assert len(result.params) == expected


@pytest.mark.parametrize(
    "columns,expected",
    [
        (['attr1'], 3),
        (['attr2'], 3),
    ],
)
def test_split_by_columns(columns, expected):
    result = sample_experiment.split_by_columns(columns)
    assert len(result) == expected


@pytest.mark.parametrize(
    "params_names_to_keep,expected_columns",
    [
        (['attr1'], ['attr1']),
        (['attr2'], ['attr2']),
        (['metric1'], ['metric1']),
        (['attr1', 'attr2'], ['attr1', 'attr2']),
        (['attr1', 'metric1'], ['attr1', 'metric1']),
    ],
)
def test_subset_params2(params_names_to_keep, expected_columns):
    result = sample_experiment.subset_params(params_names_to_keep)
    assert len(result.params.columns) == len(expected_columns)
    assert set(result.params.columns) == set(expected_columns)


@pytest.mark.parametrize(
    "runs_ids_to_keep,expected_rows",
    [
        (['run1'], ['run1']),
        (['run2'], ['run2']),
        (['run1', 'run2'], ['run1', 'run2']),
    ],
)
def test_subset_runs2(runs_ids_to_keep, expected_rows):
    result = sample_experiment.subset_runs(runs_ids_to_keep)
    assert len(result.params) == len(expected_rows)
    assert set(result.params['id'].values) == set(expected_rows)


@pytest.mark.parametrize(
    "params_names_to_drop,expected_columns",
    [
        (['attr1'], ['id', 'attr2', 'metric1', 'metric2']),
        (['attr2'], ['id', 'attr1', 'metric1', 'metric2']),
        (['metric1'], ['id', 'attr1', 'attr2', 'metric2']),
        (['attr1', 'attr2'], ['id', 'metric1', 'metric2']),
        (['attr1', 'metric1'], ['id', 'attr2', 'metric2']),
    ],
)
def test_without2(params_names_to_drop, expected_columns):
    result = sample_experiment.without(params_names_to_drop)
    assert len(result.params.columns) == len(expected_columns)
    assert set(result.params.columns) == set(expected_columns)


def create_test_experiment():
    params = pd.DataFrame({'param1': [1, 2], 'param2': [3, 4]})
    series = {'metric1': pd.DataFrame({'value': [5, 6]}), 'metric2': pd.DataFrame({'value': [7, 8]})}
    return Experiment(params, series)


def test_merge_success():
    exp1 = create_test_experiment()
    exp2 = create_test_experiment()

    merged_exp = exp1.merge(exp2)

    assert len(merged_exp.params) == 4
    assert len(merged_exp.series['metric1']) == 4
    assert len(merged_exp.series['metric2']) == 4


def test_merge_failure_different_params():
    exp1 = create_test_experiment()

    params = pd.DataFrame({'param3': [1, 2], 'param4': [3, 4]})
    series = {'metric1': pd.DataFrame({'value': [5, 6]}), 'metric2': pd.DataFrame({'value': [7, 8]})}
    exp2 = Experiment(params, series)

    with pytest.raises(ValueError) as excinfo:
        exp1.merge(exp2)

    assert "Cannot merge experiments with different parameters" in str(excinfo.value)


def test_merge_correct_dfs():
    exp1 = create_test_experiment()
    exp2 = create_test_experiment()

    merged_exp = exp1.merge(exp2)

    # Check if params DataFrame is correct
    expected_params = pd.DataFrame({'param1': [1, 2, 1, 2], 'param2': [3, 4, 3, 4]})

    pd.testing.assert_frame_equal(merged_exp.params.reset_index(drop=True), expected_params.reset_index(drop=True))

    # Check if metric1 DataFrame is correct
    expected_metric1 = pd.DataFrame({'value': [5, 6, 5, 6]})
    pd.testing.assert_frame_equal(merged_exp.series['metric1'].reset_index(drop=True),
                                  expected_metric1.reset_index(drop=True))

    # Check if metric2 DataFrame is correct
    expected_metric2 = pd.DataFrame({'value': [7, 8, 7, 8]})
    pd.testing.assert_frame_equal(merged_exp.series['metric2'].reset_index(drop=True),
                                  expected_metric2.reset_index(drop=True))
