import warnings
import gc

import pandas as pd
import pandas_profiling
import matplotlib
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, Imputer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization

from utils import draw_feature_distribution, OOFPreds, display_importances

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 200})


def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())

def missing_values_table(df):
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


def main(debug=2000):
    if debug is not False:
        rows = debug
    else:
        rows = None

    app_train = pd.read_csv('../input/application_train.csv', nrows=rows)
    app_test = pd.read_csv('../input/application_test.csv', nrows=rows)

    id_train = app_train.index
    id_test = app_test.index
    test_skid = app_test[['SK_ID_CURR']]

    # app_all = pd.concat([app_train, app_test], axis=0)
    # print('shape: {}'.format(app_all.shape))

    y = app_train['TARGET']
    ids = app_train['SK_ID_CURR']

    del app_train['SK_ID_CURR']

    # missing_values = missing_values_table(app_train)
    # print(missing_values.head(40))
    # print(missing_values.tail(10))

    profile = pandas_profiling.ProfileReport(app_train)
    rejected_variables = profile.get_rejected_variables(threshold=0.9)
    print(rejected_variables)

    features = [col for col in app_train.columns if col not in rejected_variables]
    # app_train = app_train[features]
    # app_test = app_test[features]


    # ----------------------------------------------------------------------------------------------------

    # 划分特征类型： 数值连续，数值离散， 分类

    # ----------------------------------------------------------------------------------------------------
    '''
    列名当中其实已经包含了，数据的类型

    分类：
    ---------
    TYPE CODE

    数值连续：
    ----------
    DAYS
    CNT
    RELATIVE 相对值 也就是标准化过
    AMT

    数值离散：
    ----------
    FLAG 
    RATING
    NOT  
    '''
    print('train shape {}'.format(app_train.shape))

    num_cols = app_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = app_train.select_dtypes(include=[np.object]).columns.tolist()

    print("\nnum_cols count:{}".format(len(num_cols)))
    print("\ncat_cols count:{}".format(len(cat_cols)))

    print("\nnum_cols:{}".format(num_cols))
    print("\ncat_cols:{}".format(cat_cols))

    cat_flag_cols = [col for col in cat_cols if app_train[col].nunique() == 2]
    print('\ncat flag 特征总数：{}'.format(len(cat_flag_cols)))
    print('cat flag cols {}'.format(cat_flag_cols))

    # 只有一个值
    num_1_cols = [col for col in num_cols if app_train[col].nunique() == 1]
    print('\nnum 1 特征总数：{}'.format(len(num_1_cols)))
    print('{}'.format(num_1_cols))

    num_flag_cols = [col for col in num_cols if app_train[col].nunique() == 2]
    print('\nnum flag 特征总数：{}'.format(len(num_flag_cols)))
    print('num flag cols {}'.format(num_flag_cols))

    num_discreet_cols = [col for col in num_cols if 2 < app_train[col].nunique() <= 20]
    print('\n数值离散 特征总数：{}'.format(len(num_discreet_cols)))
    print('数值离散 {}'.format(num_discreet_cols))

    num_continuous_cols = [col for col in num_cols if 20 < app_train[col].nunique()]
    print('\n数值连续 特征总数：{}'.format(len(num_continuous_cols)))
    print('数值连续 {}'.format(num_continuous_cols))

    # for col in num_cols:
    #     print(app_train[col].nunique())



    # ----------------------------------------------------------------------------------------------------

    # outlier drop

    # ----------------------------------------------------------------------------------------------------
    # print("开始清楚离群点")

    need_drop = ['AMT_INCOME_TOTAL',
                 'AMT_ANNUITY',
                 'AMT_GOODS_PRICE',
                 'APARTMENTS_AVG',
                 'BASEMENTAREA_AVG',
                 'COMMONAREA_AVG',
                 'ELEVATORS_AVG',
                 'ENTRANCES_AVG',
                 'FLOORSMAX_AVG',
                 'FLOORSMIN_AVG',
                 'YEARS_BEGINEXPLUATATION_AVG',
                 'YEARS_BEGINEXPLUATATION_MEDI',
                 'COMMONAREA_AVG',
                 'NONLIVINGAPARTMENTS_AVG',
                 'YEARS_BEGINEXPLUATATION_MODE',
                 'NONLIVINGAPARTMENTS_MODE',
                 # 'OWN_CAR_AGE'
                 ]


    train_length = app_train.shape[0]
    for col in need_drop:
        col_mean = app_train[col].mean()
        col_std = app_train[col].std()
        z = (app_train[col] - col_mean) / col_std
        outlier = z[abs(z) > 8].index

        app_train = app_train.drop(outlier)

    print('被删除的离群点个数：{}'.format(train_length - app_train.shape[0]))

    # ----------------------------------------------------------------------------------------------------

    # 数据探索可视化

    # ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    #  离群值可视化

    train_length = app_train.shape[0]
    app_all = pd.concat([app_train, app_test], axis=0)
    # for col in need_drop:
    #     draw_feature_distribution(app_all, col, 'outlier_drop')

    # ----------------------------------------------------------------------------------------------------

    # 处理异常数据 DAYS_EMPLOYED_ANOM

    # ----------------------------------------------------------------------------------------------------

    app_all['DAYS_EMPLOYED_ANOM'] = app_all["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    app_all['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # ----------------------------------------------------------------------------------------------------
    #

    fcols = 4
    frows = len(num_continuous_cols)
    plt.figure(figsize=(4 * fcols, 6 * frows))
    i = 0

    for col in num_continuous_cols:

        dat = app_train[[col, 'TARGET']].dropna()

        i += 1
        plt.subplot(frows, fcols, i)
        sns.distplot(dat[col], fit=stats.norm)
        plt.title(col + ' Original')
        plt.xlabel('')

        i += 1
        plt.subplot(frows, fcols, i)
        _ = stats.probplot(dat[col], plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[col])))
        plt.xlabel('')
        plt.ylabel('')

        i += 1
        plt.subplot(frows, fcols, i)
        trans_var = scale_minmax(dat[col].dropna())
        trans_var, lambda_var = stats.boxcox(trans_var + 1)
        trans_var = scale_minmax(trans_var)
        sns.distplot(trans_var, fit=stats.norm);
        plt.title(col + ' Tramsformed')
        plt.xlabel('')

        i += 1
        plt.subplot(frows, fcols, i)
        _ = stats.probplot(trans_var, plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(trans_var)))
        plt.xlabel('')
        plt.ylabel('')

    plt.tight_layout(h_pad=2.5)
    plt.savefig('./pic/num_continue_10000_compare.png')
    plt.show()

    no_transform = [
        'EVEVATORS_AVG',
        'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG',
        'ELEVATORS_MODE',
        'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE',
        'ELEVATORS_MEDI',
        'NONLIVINGAPARTMENTS_MEDI',
        'NONLIVINGAREA_MEDI',
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
    ]

    trans_col = [col for col in num_continuous_cols if col not in no_transform]

    for col in trans_col:
        col_trans = scale_minmax(app_all.loc[:, col])
        app_all.loc[:, col], _ = stats.boxcox(col_trans + 1)
        app_all.loc[:, col] = scale_minmax(app_all.loc[:, col])

    # 离群值可视化

    # for col in num_continuous_cols:
    #     draw_feature_distribution(app_all, col, 'outlier_boxcox')

    print("train length : {}".format(train_length))
    print('all app length : {} '.format(len(app_all)))
    app_train = app_all.iloc[:train_length, :]
    app_test = app_all.iloc[train_length:, :]

    # 对照

    fcols = 2
    frows = len(num_continuous_cols)
    plt.figure(figsize=(4 * fcols, 6 * frows))
    i = 0

    for col in num_continuous_cols:

        dat = app_train[[col, 'TARGET']].dropna()

        i += 1
        plt.subplot(frows, fcols, i)
        sns.distplot(dat[col], fit=stats.norm)
        plt.title(col + ' Original')
        plt.xlabel('')

        i += 1
        plt.subplot(frows, fcols, i)
        _ = stats.probplot(dat[col], plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[col])))
        plt.xlabel('')
        plt.ylabel('')

    plt.tight_layout(h_pad=2.5)
    plt.savefig('./pic/num_continue_boxcox.png')
    plt.show()


if __name__ == '__main__':
    main(debug=10000)