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
from sklearn.metrics import roc_auc_score, roc_curve

from bayes_opt import BayesianOptimization

from utils import draw_feature_distribution, OOFPreds, display_importances, kde_plot, skew_plot

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 200})


def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())

def missing_values_table(df):
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

    app_train = pd.read_csv('../../input/application_train.csv', nrows=rows)
    app_test = pd.read_csv('../../input/application_test.csv', nrows=rows)

    test_skid = app_test[['SK_ID_CURR']]

    cols_label = ['TARGET']
    cols_basic = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
    cols_amt = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

    cols_test = cols_basic + cols_amt + cols_label
    cols_feature = cols_basic + cols_amt

    # ----------------------------------------------------------------------------------------------------
    # 清除离群点

    print('1 清除离群点')

    app_train = app_train.loc[:, cols_test]
    app_test = app_test.loc[:, cols_test]
    origin_index = app_train.index

    train_length = app_train.shape[0]
    need_drop = ['AMT_INCOME_TOTAL']
    for col in need_drop:
        col_mean = app_train[col].mean()
        col_std = app_train[col].std()
        z = (app_train[col] - col_mean) / col_std
        outlier = z[abs(z) > 8].index
        app_train = app_train.drop(outlier)

    # print(app_train.index)
    # drop_index = [index for index in origin_index.tolist() if index not in app_train.index.tolist()]
    # print(drop_index)

    print('被删除的离群点个数：{}'.format(train_length - app_train.shape[0]))

    app_all = pd.concat([app_train, app_test], axis=0)
    print('shape: {}'.format(app_all.shape))

    missing_values = missing_values_table(app_train)
    print(missing_values)

    # ----------------------------------------------------------------------------------------------------
    print('2 构造新特征')
    # ----------------------------------------------------------------------------------------------------

    features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

    for col in features:
        app_all[col].fillna(app_all[col].dropna().median(), inplace=True)

    new_features = []
    for i in range(4):
        for j in range(i+1, 4):
            app_all[features[i] + ' / ' + features[j]] = app_all[features[i]] / app_all[features[j]]
            new_features.append(features[i] + ' / ' + features[j])

    print(new_features)

    features = cols_feature + new_features


    train_len = app_train.shape[0]
    print(train_len)
    app_train = app_all.iloc[:train_len, :]
    app_test = app_all.iloc[train_len:, :]

    # plot
    print(app_train.shape, len(features))
    kde_plot(app_train, features, 'amt_4_pic/4_orgin_persent_sum_kde.png')
    skew_plot(app_train, features, 'amt_4_pic/4_origin_persent_sum_skew.png')

    # 准备训练数据

    # ----------------------------------------------------------------------------------------------------
    print('3 准备训练数据')
    # ----------------------------------------------------------------------------------------------------

    label = app_train['TARGET']
    print('label len {}'.format(len(label)))

    if 'TARGET' in app_train:
        train = app_train.drop(columns=['TARGET'])
    else:
        train = app_train.copy()

    # Copy of the testing data
    test = app_test[features].copy()

    print(train.shape, test.shape)

    imputer = Imputer(strategy='median')
    scaler = MinMaxScaler(feature_range=(0, 1))

    imputer.fit(train)
    train = imputer.transform(train)
    test = imputer.transform(test)

    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print(train.shape[0], test.shape[0])

    train_feature = pd.DataFrame(train, columns=features, index=label.index)

    # print('label shape, train_feature shape')
    # print(label.shape, train_feature.shape)
    # print(label.head(), train_feature.head())
    # print((label.index == train_feature.index).sum())
    app_train = pd.concat([label, train_feature], axis=1)
    app_test = pd.DataFrame(test, columns=features)
    # print('app_train {}, app_test {} '.format(app_train.shape, app_test.shape))

    # plot

    kde_plot(app_train, features, 'amt_4_pic/4_amt_persent_fillna_scaler_sum_kde.png')
    skew_plot(app_train, features, 'amt_4_pic/4_amt_persent_fillna_scaler_sum_skew.png')

    # 模型训练

    print('4 lgb 训练')

    params_lgb = {'nthread': 4,
                  'n_estimators': 10000,
                  'learning_rate': 0.02,
                  'num_leaves': 34,
                  'colsample_bytree': 0.9497036,
                  'subsample': 0.8715623,
                  'max_depth': 8,
                  'reg_alpha': 0.041545473,
                  'reg_lambda': 0.0735294,
                  'min_split_gain': 0.0222415,
                  'min_child_weight': 39.3259775,
                  'silent': -1,
                  'verbose': -1, }

    oof_preds, sub_preds, feature_importance, metrics = OOFPreds(train_feature, label, app_test, params_lgb, clf='lgb')

    # display_importances(feature_importance, num_features=4, filename='./basic_4_pic/feature_importance.png')

    print(metrics)

    sub_preds = pd.concat([test_skid, sub_preds], axis=1)

    sub_preds.to_csv('lgb_4amt_persent_-fillmedian-minmax-sum-val-180628.csv', index=False)

    # ----------------------------------------------------------------------------------------------------
    print('5 box_cox')
    # ----------------------------------------------------------------------------------------------------

    train_len = app_train.shape[0]
    app_all = pd.concat([app_train, app_test], axis=0)
    print(train_len)

    for col in features:
        col_trans = scale_minmax(app_all.loc[:, col])
        app_all.loc[:, col], _ = stats.boxcox(col_trans + 1)
        app_all.loc[:, col] = scale_minmax(app_all.loc[:, col])

    app_train = app_all.iloc[:train_len, :]
    app_test = app_all.iloc[train_len:, :]

    kde_plot(app_train, features, 'amt_4_pic/4_persent_boxcox_sum_kde.png')
    skew_plot(app_train, features, 'amt_4_pic/4_persent_boxcox_sum_skew.png')

    # ----------------------------------------------------------------------------------------------------
    print('6 lgb 训练')
    # ----------------------------------------------------------------------------------------------------

    params_lgb = {'nthread': 4,
                  'n_estimators': 10000,
                  'learning_rate': 0.02,
                  'num_leaves': 34,
                  'colsample_bytree': 0.9497036,
                  'subsample': 0.8715623,
                  'max_depth': 8,
                  'reg_alpha': 0.041545473,
                  'reg_lambda': 0.0735294,
                  'min_split_gain': 0.0222415,
                  'min_child_weight': 39.3259775,
                  'silent': -1,
                  'verbose': -1, }


    print(app_train.shape, app_test.shape)
    train_feature = app_train[features]
    test = app_test[features]

    oof_preds, sub_preds, feature_importance, metrics = OOFPreds(train_feature, label, test, params_lgb, clf='lgb')

    # print(feature_importance.groupby())
    display_importances(feature_importance, num_features=15, filename='feature_importance_sum.png')

    print(metrics)

    fpr, tpr, _ = roc_curve(label, oof_preds)

    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr,
        tpr,
        lw=1,
        alpha=0.3,
        label='ROC (AUC = %0.4f ' % (metrics.iloc[5, 2])
        )

    plt.plot(
        [0, 1], [0, 1],
        linestyle='--',
        lw=2,
        color='r',
        label='Luck',
        alpha=.8)
    plt.savefig('roc_persent_sum.png')

    sub_preds = pd.concat([test_skid, sub_preds], axis=1)

    sub_preds.to_csv('lgb_4amt_persent-boxcox-sum-val-180628.csv', index=False)


if __name__ == '__main__':
    main(debug=False)