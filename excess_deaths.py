import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

start_date = datetime.date(2020, 2, 1)


def get_dates(row, start_col="start_date", end_col="end_date"):
    year = int(row[start_col].split('-')[0])
    month = int(row[start_col].split('-')[1])
    year_end = int(row[end_col].split('-')[0])
    month_end = int(row[end_col].split('-')[1])
    if (year, month) != (year_end, month_end):
        print("Start and end month are not the same:")
        print(row)
    return pd.Series([year, month, str("{}/{}".format(month, year-int(year/100)*100))],
                     index=["year", "month", "month_str"])


def start_plot():
    """Function to start a plot.
    """
    plt.figure(figsize=(8, 5))


def end_plot(**kwargs):
    """Function to finish a plot.

    Args:
        title (str): Title for the plot.
        xlabel (str): X axis label.
        ylabel (str): Y axis label.
        log_y (bool): Uses logarithmic scale on the y axis.
            Defaults to False.
        xlim (tuple): Sets the plot's x limits.
        ylim (tuple): Sets the plot's y limits.
        zero_y (bool): Sets the lower y limit to zero.
        legend (bool): Prints the legend. Defaults to False.
        legend_args (dict): Keyword arguments for the plt.legend()
            function.
        tight_layout (bool): Sets tight layout to pad=0.5.
            Defatlts to True.
        filename (str): File name to save plot, if not given only
            shows plot.
    """
    title = kwargs.get('title', None)
    if title is not None:
        plt.title(title)
    xlabel = kwargs.get('xlabel', False)
    if xlabel:
        plt.xlabel(xlabel)
    ylabel = kwargs.get('ylabel', False)
    log_y = kwargs.get('log_y', False)
    if log_y:
        if ylabel:
            ylabel = ylabel+" (log)"
        plt.yscale("log")
    if ylabel:
        plt.ylabel(ylabel)
    xlim = kwargs.get('xlim', None)
    if xlim is not None:
        plt.xlim(xlim)
    ylim = kwargs.get('ylim', None)
    if ylim is not None:
        plt.ylim(ylim)
    zero_y = kwargs.get('zero_y', False)
    if zero_y:
        ax = plt.gca()
        plt.ylim((0, ax.get_ylim()[1]))
    legend = kwargs.get('legend', False)
    legend_args = kwargs.get('legend_args', {})
    if legend or legend_args != {}:
        plt.legend(**legend_args)
    tight_layout = kwargs.get('tight_layout', True)
    if tight_layout:
        plt.tight_layout(pad=0.5)
    filename = kwargs.get('filename', None)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


class excess_deaths:
    def __init__(self, df, start_date=start_date, mask=slice(None),
                 end_date=None):
        """Initilizaes an excess_deths objects and organizes data.

        Args:
            df (DataFrame): DataFrame with deaths data, required columns:
                start_date, end_date, deaths_total.
            start_date ([type], optional): Date to divide data, deaths
                before start_date will be considered in the mean deaths.
                Defaults to "2020-2-1".
            mask (Series, optional): A boolean mask to select rows in the
                DataFrame. Defaults to slice(None).
        """
        df_sel = df[mask & df.deaths_total.notnull()]
        df_sel = pd.concat([df_sel,
                            df_sel.apply(get_dates, axis=1,
                                         result_type='expand')],
                           axis=1)

        # Process period
        self.start_month = start_date.month
        self.start_year = start_date.year

        if type(end_date) is datetime.date:
            self.end_month = end_date.month
            self.end_year = end_date.year
        else:
            self.end_year = self.df_sel.year.max()
            self.end_month = self.df_sel[self.df_sel.year == self.end_year].month.max()

        if self.start_year == self.end_year:
            n_months = self.end_month = self.start_month
        else:
            n_months = (13 - self.start_month) \
                       + 12*(self.end_year - self.start_year - 1) \
                       + self.end_month

        # Separate data
        prev_mask = (df_sel.year < self.start_year) | ((df_sel.year == self.start_year)
                                               & (df_sel.month < self.start_month))
        self.df_sel = df_sel[~prev_mask]
        self.sel_sum = \
            self.df_sel.groupby(["month_str", "month", "year"]).sum()
        self.prev_sum = \
            df_sel[prev_mask].groupby(["month_str", "month", "year"]).sum()

        # Check if we have enough data
        if np.sum(prev_mask) < 12 or np.sum(~prev_mask) < n_months:
            raise RuntimeError("Not enough data!")
        if len(self.prev_sum.groupby("month").sum()) < 12:
            raise RuntimeError("Not enough data!")

    def plot_deaths(self, **plt_args):
        """Plots the mean deaths curve and the deaths after start_date.

        Args:
            **plt_args: See end_plot parameters.
        """
        # Collects data
        months = {}
        deaths = {}
        # Iterate in all months and years after start_date
        # collecting the number of deaths.
        for year in self.df_sel.year.unique():
            df_year = self.df_sel[self.df_sel.year == year]
            months[year] = df_year.month.unique()
            if year == self.end_year:
                months[year] = [m for m in months[year] if m <= self.end_month]
            deaths[year] = np.zeros(len(months[year]))
            for i, month in enumerate(months[year]):
                deaths[year][i] = \
                    df_year[df_year.month == month].deaths_total.sum()

        # Plots data
        start_plot()
        # Plot mean deaths
        sns.lineplot(x="month", y="deaths_total", data=self.prev_sum)
        line = plt.gca().lines[-1]
        line.set_label("mean jan/2015-jan/2020")
        # Plots deaths after start_date
        for year in deaths.keys():
            plt.plot(months[year], deaths[year], label=year)
        plt.xticks(np.arange(1, 13), ["jan", "feb", "mar", "apr", "may", "jun",
                                     "jul", "aug", "sep", "oct", "nov", "dec"])
        if "legend" not in plt_args:
            plt_args['legend'] = True
        if "xlabel" not in plt_args:
            plt_args["xlabel"] = "Month"
        if "ylabel" not in plt_args:
            plt_args["ylabel"] = "Total deaths"
        end_plot(**plt_args)

    def count_excess_deaths(self):
        """[summary]

        Args:
            end_date (datetime.date, optional): Last month to count
                excess deaths. Defaults to None.

        Returns:
            float, float: Number of deaths above the mean and above
                the mean plus the standard deviation.
        """
        # Get previous years data
        mean_deaths = self.prev_sum.groupby("month").mean().deaths_total
        std_deaths = self.prev_sum.groupby("month").std().deaths_total

        # Count excess deaths
        count = 0
        mean_count = 0
        count_std = 0
        for (_, month, year), row in self.sel_sum.iterrows():
            if (year > self.end_year) \
               or (year == self.end_year and month > self.end_month):
                continue
            count += row.deaths_total - mean_deaths[month]
            mean_count += mean_deaths[month]
            count_std += row.deaths_total - (mean_deaths[month]
                                             + std_deaths[month])

        return count_std, count, mean_count

    def plot_p_score(self, end_date=None, **plt_args):
        """Plots the percentage above the mean number of deaths
            (p-score, see
            https://ourworldindata.org/excess-mortality-covid#how-is-excess-mortality-measured).

        Args:
            end_date (datetime.date, optional): Last month to count
                excess deaths. Defaults to None.
        """
        # Get previous years data
        mean_deaths = self.prev_sum.groupby("month").mean().deaths_total

        # Calculate p-score for each month between start_date and end_date
        p_score = {}
        for (month_str, month, year), row in self.sel_sum.iterrows():
            if (year > self.end_year) \
               or (year == self.end_year and month > self.end_month):
                continue
            p_score[(year, month, month_str)] = 100 * (row.deaths_total
                                                       - mean_deaths[month]) \
                                                    / mean_deaths[month]

        # Sort and aggregate results
        y = []
        x = []
        for key in sorted(p_score.keys()):
            y.append(p_score[key])
            x.append(key[2])

        # Plot data
        start_plot()
        plt.plot(np.arange(len(x)), y)
        if "title" not in plt_args:
            plt_args["title"] = "Excess deaths"
        if "xlabel" not in plt_args:
            plt_args["xlabel"] = "Month"
        if "ylabel" not in plt_args:
            plt_args["ylabel"] = "Percentage above mean deaths"
        plt.xticks(np.arange(len(x)), x)
        end_plot(**plt_args)

    def aggregate_and_sort_deaths(self, df_group, mean_deaths=None, deaths_col="deaths_total"):
        if mean_deaths is None:
            mean_deaths = np.zeros(13)
        # Calculate total deaths for each month between start_date and end_date
        deaths = {}
        for (month_str, month, year), row in df_group.iterrows():
            if (year > self.end_year) \
               or (year == self.end_year and month > self.end_month):
                continue
            deaths[(year, month, month_str)] = row[deaths_col] - mean_deaths[month]

        # Sort and aggregate results
        y = []
        x = []
        for key in sorted(deaths.keys()):
            y.append(deaths[key])
            x.append(key[2])
        cum_y = []
        for i in range(len(y)):
            cum_y.append(np.sum(y[:i+1]))

        return x, cum_y

    def plot_excess_deaths(self, extra_data=None, **plt_args):
        """Plots the percentage above the mean number of deaths
            (p-score, see
            https://ourworldindata.org/excess-mortality-covid#how-is-excess-mortality-measured).

        Args:
            end_date (datetime.date, optional): Last month to count
                excess deaths. Defaults to None.
        """
        # Get previous years data
        mean_deaths = self.prev_sum.groupby("month").mean().deaths_total

        # Calculate deaths for each month between start_date and end_date
        x, y = self.aggregate_and_sort_deaths(self.sel_sum, mean_deaths)

        # Plot data
        start_plot()
        plt.plot(np.arange(len(x)), y, label="Excess deaths")

        # Plots extra data and difference
        if extra_data is not None:
            extra_y = extra_data['y']
            extra_x = [x.index(x_str) for x_str in extra_data['x']]
            plt.plot(extra_x, extra_data['y'], label=extra_data['label'])
            plt.ylabel("Deaths")
            plt.xlabel("Month")
            ax = plt.gca()
            ax.legend()
            if plt_args.get("zero_y", False):
                plt.ylim((0, ax.get_ylim()[1]))
            ax2 = ax.twinx()
            perc_dif = []
            perc_x = []
            for i, x_str in enumerate(extra_data['x']):
                if extra_y[i] > 0:
                    perc = 100*(y[x.index(x_str)] - extra_y[i])/extra_y[i]
                    perc_dif.append(perc)
                    perc_x.append(x.index(x_str))
            ax2.plot(perc_x, perc_dif, label="Difference", color='r')
            ax2.legend()
            plt.ylabel("Difference percentage")

        if "title" not in plt_args:
            plt_args["title"] = "Excess deaths"
        if "xlabel" not in plt_args:
            plt_args["xlabel"] = "Month"
        plt.xticks(np.arange(len(x)), x)
        end_plot(**plt_args)

    def plot_excess_deaths_comparison(self, compare_df, date_col, deaths_col,
                                       label, **plt_args):
        """Plots a comparison between the excess deaths curve
            and another death curve.

        Args:
            compare_df (DataFrame): DataFrame with another deaths data.
            date_col (str): Column containing the date.
            deaths_col (str): Column containing the number of deaths.
            label (str): Label for the plot.
        """
        # Load data
        compare_df = pd.concat(
            [
                compare_df,
                compare_df.apply(get_dates, axis=1, result_type='expand',
                                 start_col=date_col, end_col=date_col)
            ],
            axis=1
        )

        x, y = self.aggregate_and_sort_deaths(
                    compare_df.groupby(["month_str", "month", "year"]).sum(),
                    deaths_col=deaths_col)

        data = {"x": x, "y": y, "label": label}
        self.plot_excess_deaths(extra_data=data, **plt_args)
