import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return newFunc


def kde_target(var_name, df):
    """用于展示单一特征中，label分布的函数"""
    corr = df['TARGET'].corr(df[var_name])

    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize=(12, 6))

    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label='TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label='TARGET == 1')

    # 我更喜欢用 distplot
    # sns.distplot(df.ix[df['TARGET'] == 0, var_name], label='TARGET == 0')
    # sns.distplot(df.ix[df['TARGET'] == 1, var_name], label='TARGET == 1')

    # label the plot
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend()
    plt.show()

    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)

@exeTime
def agg_numeric(df, group_var, df_name):
    """聚合df的数值类型特征.

    Parameters
    --------
        df (dataframe):
            需要统计聚合的 dataframe

        group_var : string
            分组的依照变量，通常是 id 值
            The variable by which to group the dataframe. For each unique
            value of this variable, the final dataframe will have one row

        df_name (string):
            给新特征列重命名的 前缀
            Variable added to the front of column names to keep track of columns

    Return
    --------
        agg (dataframe):
            已经统计聚合过的dataframe
            针对所有数值型特征， 统计 mean min max sum
            列已经被重命名过


    """

    # First calculate counts
    counts = pd.DataFrame(df.groupby(group_var, as_index=False)[df.columns[1]].count()).rename(
        columns={df.columns[1]: '%s_counts' % df_name})

    # Group by the specified variable and calculate the statistics
    agg = df.groupby(group_var).agg(['mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    #  Rename the columns
    agg.columns = columns

    # Merge with the counts
    agg = agg.merge(counts, on=group_var, how='left')

    return agg


def target_corrs(df):
    # List of correlations
    corrs = []

    # Iterate through the columns
    for col in df.columns:
        print(col)
        # Skip the target column
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df[col])

            # Append the list as a tuple
            corrs.append((col, corr))

    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)

    return corrs


@exeTime
def count_categorical(df, group_var, df_name):
    """针对分类型数据，统计聚合 每个特征不同变量的计数和占比

    Parameters
    --------
    df : dataframe
        需要统计聚合的 dataframe

    group_var : string
        分组的依照变量，通常是 id 值
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row

    df_name : string
        给新特征列重命名的 前缀
        Variable added to the front of column names to keep track of columns


    Return
    --------
    categorical : dataframe
        已经统计聚合过分类特征的的dataframe，每一行只对应一个 group_var 的值


    """

    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    return categorical


def missing_values_table(df):
    """"用于计算每个特征的空值的函数"""
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns