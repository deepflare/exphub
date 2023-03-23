class Selector:

    def __init__(self, dataframe, metric):
        """
        Initialize the Selector class with a DataFrame and a metric.

        Parameters:
            dataframe (pd.DataFrame): Dataframe to select from
            metric (function): a function that returns a value for each row of the DataFrame

        """
        self.df = dataframe
        self.metric = metric

    def filter_rows(self, condition):
        """
        Filters the rows of the DataFrame based on a given condition.

        Parameters:
            condition (function): a function that returns a boolean value for each row of the DataFrame

        """
        self.df = self.df[self.df.apply(condition, axis=1)]

    def top_k(self, k):
        """
        Select the top-k rows of the DataFrame based on the given metric.

        Parameters:
            k (int): number of rows to select

        Returns:
            pd.DataFrame: The top k rows of the DataFrame based on the given metric
        """
        self.df['metric_value'] = self.df.apply(self.metric, axis=1)
        return self.df.nlargest(k, 'metric_value')

    def top_k_multiple(self, k, metric_list):
        """
        Select the top-k rows of the DataFrame based on multiple metrics.

        Parameters:
            k (int): number of rows to select
            metric_list (list): a list of functions that return a value for each row of the DataFrame

        Returns:
            pd.DataFrame: The top k rows of the DataFrame based on the overall score

        """
        for metric in metric_list:
            self.df[metric.__name__] = self.df.apply(metric, axis=1)
        self.df['overall_score'] = self.df[[metric.__name__ for metric in metric_list]].sum(axis=1)
        return self.df.nlargest(k, 'overall_score')

    def return_format(self, format_type):
        """
        Return the selected rows in a specific format.

        Parameters:
            format_type (str): the format to return the selected rows in.

        Returns:
            depending on the format type it will return the selected rows in that format

        """
        if format_type == "list_of_dict":
            return self.df.to_dict("records")
        elif format_type == "json":
            return self.df.to_json()

    def return_columns(self, columns_list):
        """
        Return the selected rows with specific columns.

        Parameters:
            columns_list (list): a list of strings representing the names of the columns to select

        Returns:
            pd.DataFrame: The selected rows with the specified columns

        """
        self.df = self.df[columns_list]
        return self.df
