import pandas as pd
from typing import List, Any
import functools

import functools
import pandas as pd
from typing import Any, List


class Aggregator:
    """A class that represents an aggregator function to be applied to a DataFrame.

    An Aggregator object is a callable that takes a DataFrame and an optional inplace
    parameter and returns the modified DataFrame with a new column appended to it. The
    new column's values are computed using the aggregator function specified in the
    Aggregator object's constructor.
    """

    def __init__(self, fn, label: Any) -> None:
        """Constructs an Aggregator object.

        Args:
            fn (function): The aggregator function that will be applied to the
                DataFrame. The function should take a DataFrame as input and return
                a Series or a DataFrame.
            label (Any): The label to be used for the new column in the DataFrame.
        """
        self.fn = fn
        self.label = label

    def __call__(self, incoming_df: pd.DataFrame, inplace: bool = False) -> Any:
        """Modifies a DataFrame by adding a new column computed using the aggregator function.

        Args:
            incoming_df (pd.DataFrame): The DataFrame to be modified.
            inplace (bool, optional): If True, the incoming DataFrame will be modified in place.
                If False, a copy of the DataFrame will be created and modified. Defaults to False.

        Returns:
            Any: The modified DataFrame.
        """
        df = incoming_df.copy() if not inplace else incoming_df
        df[self.label] = self.fn(df)
        return df


class AggregatorChain:
    """A class that represents a chain of Aggregator objects.

    An AggregatorChain object is a callable that takes a DataFrame and an optional
    inplace parameter and returns the modified DataFrame after applying all the
    Aggregator objects in the chain, in the order they were added.
    """

    def __init__(self, aggs: List[Aggregator]) -> None:
        """Constructs an AggregatorChain object.

        Args:
            aggs (List[Aggregator]): A list of Aggregator objects to be applied in the chain.
        """
        self.aggs = aggs

    def __call__(self, df: pd.DataFrame, inplace=False) -> Any:
        """Modifies a DataFrame by applying a chain of Aggregator objects.

        Args:
            df (pd.DataFrame): The DataFrame to be modified.
            inplace (bool, optional): If True, the incoming DataFrame will be modified in place.
                If False, a copy of the DataFrame will be created and modified. Defaults to False.

        Returns:
            Any: The modified DataFrame.
        """
        df_n = df.copy() if not inplace else df
        return functools.reduce(lambda x, y: y(x, inplace=False), self.aggs, df_n)

    def labels(self) -> List[str]:
        """Returns a list of labels for the new columns added"""

        return list(map(lambda x: x.label, self.aggs))

    def __add__(self, other: 'AggregatorChain') -> 'AggregatorChain':
        """Adds another AggregatorChain object to the current one.

        Args:
            other (AggregatorChain): The AggregatorChain object to be added.

        Returns:
            AggregatorChain: A new AggregatorChain object that is the result of adding the two
                AggregatorChain objects.
        """
        return AggregatorChain(self.aggs + other.aggs)


class Vault:
    """A class that stores a collection of predefined Aggregator chains.

    The Vault class provides a convenient way to retrieve predefined
    Aggregator chains by name. The available Aggregator chains are stored as
    class attributes.
    """
    MEAN = AggregatorChain([Aggregator(lambda x: x.mean(axis=1, numeric_only=True), 'mean')])
    _NO_AGGR = AggregatorChain([Aggregator(lambda x: x.mean(axis=1, numeric_only=True), 'no_aggr')])
    MEAN_STD = AggregatorChain([
        Aggregator(lambda df: df.mean(axis=1, numeric_only=True), 'mean'),
        Aggregator(lambda df: df.mean(axis=1, numeric_only=True) - df.std(axis=1), 'mean_std_minus'),
        Aggregator(lambda df: df.mean(axis=1, numeric_only=True) + df.std(axis=1), 'mean_std_plus'),
    ])
    MIN = AggregatorChain([Aggregator(lambda x: x.min(axis=1, numeric_only=True), 'min')])
    MAX = AggregatorChain([Aggregator(lambda x: x.max(axis=1, numeric_only=True), 'max')])
    SUM = AggregatorChain([Aggregator(lambda x: x.sum(axis=1, numeric_only=True), 'sum')])
    MEDIAN = AggregatorChain([Aggregator(lambda x: x.median(axis=1, numeric_only=True), 'median')])
    VAR = AggregatorChain([Aggregator(lambda x: x.var(axis=1, numeric_only=True), 'var')])
    STD = AggregatorChain([Aggregator(lambda x: x.std(axis=1, numeric_only=True), 'std')])
