"""Data utility functions.

NAME: Nick Wunderle
DATE: Fall 2023
CLASS: CPSC 322

"""

from math import sqrt

from data_table import DataTable, DataRow
import matplotlib.pyplot as plt


#----------------------------------------------------------------------
# HW5
#----------------------------------------------------------------------

def normalize(table, column):
    """Normalize the values in the given column of the table. This
    function modifies the table.

    Args:
        table: The table to normalize.
        column: The column in the table to normalize.

    """
    max_value = max(column_values(table,column))
    min_value = min(column_values(table, column))
    row_index = 0
    for row in table:
        table.update(row_index, column, (row[column]-min_value)/(max_value-min_value))
        row_index+=1


def discretize(table, column, cut_points):
    """Discretize column values according to the given list of n-1
    cut_points to form n ordinal values from 1 to n. This function
    modifies the table.

    Args:
        table: The table to discretize.
        column: The column in the table to discretize.

    """
    row_index = 0
    cut_points.sort()
    for row in table:
        value = row[column]
        bin_number = 1
        for cut_point in cut_points:
            if value < cut_point:
                break
            bin_number += 1
        table.update(row_index, column, bin_number)
        row_index+=1


#----------------------------------------------------------------------
# HW4

#----------------------------------------------------------------------


def column_values(table, column):
    """Returns a list of the values (in order) in the given column.

    Args:
        table: The data table that values are drawn from
        column: The column whose values are returned
    
    """
    values = []
    for row in table:
        values.append(row[column])
    return values


def mean(table, column):
    """Returns the arithmetic mean of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the mean from

    Notes: 
        Assumes there are no missing values in the column.

    """
    values_list = column_values(table, column)
    list_sum = 0
    for i in values_list:
        list_sum += i
    return list_sum/len(values_list)


def variance(table, column):
    """Returns the variance of the values in the given table column.

    Args:
        table: The data table that values are drawn from
        column: The column to compute the variance from

    Notes:
        Assumes there are no missing values in the column.

    """
    values_list = column_values(table, column)
    list_mean = mean(table, column)
    list_sum = 0
    for i in values_list:
        list_sum += (i-list_mean)**2
    return list_sum/len(values_list)


def std_dev(table, column):
    """Returns the standard deviation of the values in the given table
    column.

    Args:
        table: The data table that values are drawn from
        column: The colume to compute the standard deviation from

    Notes:
        Assumes there are no missing values in the column.

    """
    column_variance = variance(table,column)
    return sqrt(column_variance)


def covariance(table, x_column, y_column):
    """Returns the covariance of the values in the given table columns.
    
    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x-values"
        y_column: The column with the "y-values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    x_values = column_values(table, x_column)
    y_values = column_values(table, y_column)
    x_mean = mean(table, x_column)
    y_mean = mean(table, y_column)
    total_sum = 0
    for i in range(len(x_values)):
        total_sum += (x_values[i]-x_mean)*(y_values[i]-y_mean)
    return total_sum/len(x_values)


def linear_regression(table, x_column, y_column):
    """Returns a pair (slope, intercept) resulting from the ordinary least
    squares linear regression of the values in the given table columns.

    Args:
        table: The data table that values are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    """
    slope = (covariance(table, x_column, y_column)*len(column_values(table, x_column)))/(variance(table, x_column)*len(column_values(table, x_column)))
    return slope, mean(table, y_column)-slope*mean(table, x_column)


def correlation_coefficient(table, x_column, y_column):
    """Return the correlation coefficient of the table's given x and y
    columns.

    Args:
        table: The data table that value are drawn from
        x_column: The column with the "x values"
        y_column: The column with the "y values"

    Notes:
        Assumes there are no missing values in the columns.        

    """
    columns_covariance = covariance(table, x_column, y_column)
    x_std = std_dev(table, x_column)
    y_std = std_dev(table, y_column)
    return columns_covariance/(x_std*y_std)


def frequency_of_range(table, column, start, end):
    """Return the number of instances of column values such that each
    instance counted has a column value greater or equal to start and
    less than end. 
    
    Args:
        table: The data table used to get column values from
        column: The column to bin
        start: The starting value of the range
        end: The ending value of the range

    Notes:
        start must be less than end

    """
    count = 0
    values_list = column_values(table, column)
    for val in values_list:
        if val >= start and val < end:
            count +=1
    return count


def histogram(table, column, nbins, xlabel, ylabel, title, filename=None):
    """Create an equal-width histogram of the given table column and number of bins.
    
    Args:
        table: The data table to use
        column: The column to obtain the value distribution
        nbins: The number of equal-width bins to use
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()
    
    plt.grid(axis = 'y', color = '0.85', zorder = 0)
    
    plt.hist(column_values(table, column), bins = nbins, alpha = 0.5, color = 'b', rwidth = 0.8, zorder = 3)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if filename:
        plt.savefig(filename, format = 'svg')
    else:
        plt.show()
    

def scatter_plot_with_best_fit(table, xcolumn, ycolumn, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values that includes the "best fit" line.
    
    Args:
        table: The data table to use
        xcolumn: The column for x-values
        ycolumn: The column for y-values
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    slope, intercept = linear_regression(table, xcolumn, ycolumn)
    y_fit = [slope*x+intercept for x in column_values(table, xcolumn)]
    
    plt.figure()
    
    plt.grid(color = '0.85', zorder = 0)
    #scatter plot
    plt.plot(column_values(table, xcolumn), column_values(table, ycolumn), color = 'b', marker = '.', alpha = 0.2,
             markersize = 16, linestyle = '', zorder = 3)
    #line of best fit
    plt.plot(column_values(table, xcolumn), y_fit, color = 'g')
    #y mean
    plt.axhline(y=mean(table, ycolumn), color ='r', linestyle = '--', alpha = 0.4)
    #x mean
    plt.axvline(x=mean(table, xcolumn), color ='r', linestyle = '--', alpha = 0.4)
    plt.text(.85, .9, 'r = ' "%.2f" % correlation_coefficient(table, xcolumn, ycolumn), color ='r', transform=plt.gca().transAxes)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if filename:
        plt.savefig(filename, format = 'svg')
    else:
        plt.show()

    

#----------------------------------------------------------------------
# HW3
#----------------------------------------------------------------------

def distinct_values(table, column):
    """Return the unique values in the given column of the table.
    
    Args:
        table: The data table used to get column values from.
        column: The column of the table to get values from.

    Notes:
        Returns a list of unique values
    """
    check_set = set()
    for row in table:
        check_set.add(row[column])
    return list(check_set)


def remove_missing(table, columns):
    """Return a new data table with rows from the given one without
    missing values.

    Args:
        table: The data table used to create the new table.
        columns: The column names to check for missing values.

    Returns:
        A new data table.

    Notes: 
        Columns must be valid.

    """
    for col in columns:
        if col not in table.columns():
            raise ValueError('no such column')
    new_table = DataTable(table.columns())
    for row in table:
        missing = False
        for col in columns:
            if row[col] == '':
               missing = True
               break       
        if not missing:
            new_table.append(row.values())
    return new_table


def duplicate_instances(table):
    """Returns a table containing duplicate instances from original table.
    
    Args:
        table: The original data table to check for duplicate instances.

    """
    new_table = DataTable(table.columns())
    unique_instances = set()
    new_table_unique = set()
    for row in table:
        #check for dupes
        if tuple(row.values()) in unique_instances:
            #check for multiple dupes
            if tuple(row.values()) not in new_table_unique:
                new_table.append(row.values())
                new_table_unique.add(tuple(row.values()))
        else:
            unique_instances.add(tuple(row.values()))
    return new_table
        

                    
def remove_duplicates(table):
    """Remove duplicate instances from the given table.
    
    Args:
        table: The data table to remove duplicate rows from

    """
    new_table = table.copy()
    unique_instances = set()
    row_index = 0
    while row_index < new_table.row_count():
        row = new_table[row_index]
        row_values = tuple(row.values())
        if row_values in unique_instances:
            del new_table[row_index]
        else:
            unique_instances.add(row_values)
            row_index += 1
    return new_table


def partition(table, columns):
    """Partition the given table into a list containing one table per
    corresponding values in the grouping columns.
    
    Args:
        table: the table to partition
        columns: the columns to partition the table on
    """
    partition_values = {}
    for row in table:
        key = tuple(row.values(columns))
        if key in partition_values:
            partition_values[key].append(row.values())
        else:
            partition_values[key] = DataTable(table.columns())
            partition_values[key].append(row.values())
    return list(partition_values.values())


def summary_stat(table, column, function):
    """Return the result of applying the given function to a list of
    non-empty column values in the given table.

    Args:
        table: the table to compute summary stats over
        column: the column in the table to compute the statistic
        function: the function to compute the summary statistic

    Notes: 
        The given function must take a list of values, return a single
        value, and ignore empty values (denoted by the empty string)

    """
    list_values = []
    for row in table:
        if row[column] != '':
            list_values.append(row[column])
    return function(list_values)

def replace_missing(table, column, partition_columns, function):
    """Replace missing values in a given table's column using the provided
     function over similar instances, where similar instances are
     those with the same values for the given partition columns.

    Args:
        table: the table to replace missing values for
        column: the coumn whose missing values are to be replaced
        partition_columns: for finding similar values
        function: function that selects value to use for missing value

    Notes: 
        Assumes there is at least one instance with a non-empty value
        in each partition

    """ 
    new_table = DataTable(table.columns())
    for row in table:
        value = row[column]
        if value == "":
            similar_instances = [r for r in table if all(r[c] == row[c] for c in partition_columns)]
            if similar_instances:
                non_empty_values = [r[column] for r in similar_instances if r[column] != ""]
                if non_empty_values:
                    selected_value = function(non_empty_values)
                    row[column] = selected_value
        new_table.append(row.values())
    return new_table
    
    
def summary_stat_by_column(table, partition_column, stat_column, function):
    """Returns for each partition column value the result of the statistic
    function over the given statistics column.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups from
        stat_column: the column to compute the statistic over
        function: the statistic function to apply to the stat column

    Notes:
        Returns a list of the groups and a list of the corresponding
        statistic for that group.

    """
    # TODO
    pass


def frequencies(table, partition_column):
    """Returns for each partition column value the number of instances
    having that value.

    Args:
        table: the table for computing the result
        partition_column: the column used to create groups

    Notes:

        Returns a list of the groups and a list of the corresponding
        instance count for that group.

    """
    group_counts = {}
    for row in table:
        group_value = row[partition_column]
        if group_value not in group_counts:
            group_counts[group_value] = 1
        else:
            group_counts[group_value] += 1
    
    groups = list(group_counts.keys())
    counts = [group_counts[group] for group in groups]
    
    return groups, counts


def dot_chart(xvalues, xlabel, title, filename=None):
    """Create a dot chart from given values.
    
    Args:
        xvalues: The values to display
        xlabel: The label of the x axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # reset figure
    plt.figure()
    # dummy y values
    yvalues = [1] * len(xvalues)
    # create an x-axis grid
    plt.grid(axis='x', color='0.85', zorder=0)
    # create the dot chart (with pcts)
    plt.plot(xvalues, yvalues, 'b.', alpha=0.2, markersize=16, zorder=3)
    # get rid of the y axis
    plt.gca().get_yaxis().set_visible(False)
    # assign the axis labels and title
    plt.xlabel(xlabel)
    plt.title(title)
    # save as file or just show
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    # close the plot
    plt.close()

    
def pie_chart(values, labels, title, filename=None):
    """Create a pie chart from given values.
    
    Args:
        values: The values to display
        labels: The label to use for each value
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.gca().get_yaxis().set_visible(False)
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plt.show()
    

def bar_chart(bar_values, bar_names, xlabel, ylabel, title, filename=None):
    """Create a bar chart from given values.
    
    Args:
        bar_values: The values used for each bar
        bar_labels: The label for each bar value
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()
    plt.grid(axis = 'y', color = '0.85 ', zorder = 0)
    plt.bar(bar_values, bar_names, align = 'center', zorder = 3)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if filename:
        plt.savefig(filename, format = 'svg')
    else:
        plt.show()

    
def scatter_plot(xvalues, yvalues, xlabel, ylabel, title, filename=None):
    """Create a scatter plot from given values.
    
    Args:
        xvalues: The x values to plot
        yvalues: The y values to plot
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    plt.figure()
    plt.grid(color ='0.85 ', zorder =0)
    plt.plot(xvalues, yvalues, color='b', marker='.', alpha=0.2,
             markersize=16, linestyle='', zorder=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if filename:
        plt.savefig(filename, format='svg')
    else:
        plot.show()

def box_plot(distributions, labels, xlabel, ylabel, title, filename=None):
    """Create a box and whisker plot from given values.
    
    Args:
        distributions: The distribution for each box
        labels: The label of each corresponding box
        xlabel: The label of the x-axis
        ylabel: The label of the y-axis
        title: The title of the chart
        filename: Filename to save chart to (in SVG format)

    Notes:
        If filename is given, chart is saved and not displayed.

    """
    # TODO

    
    
