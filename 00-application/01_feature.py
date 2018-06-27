import warnings
import gc

import pandas as pd
import matplotlib
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, Imputer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

from utils import draw_feature_distribution, OOFPreds, display_importances

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 200})


def kfold(x, y, x_test, num_folds, features, stratified=False):
    """用于评估模型的函数
    Args:
        x, y, x_test：数据
        num_folds：k折交叉验证
        features: 特征
        stratified：是否根据label进行分层抽样
    Return:


    Raises:

    """
    # Divide in training/validation and test data

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(x.shape, x_test.shape))
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(x.shape[0])
    sub_preds = np.zeros(x_test.shape[0])
    feature_importance_df = pd.DataFrame()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x, y)):
        train_x, train_y = x[train_idx, :], y[train_idx]
        valid_x, valid_y = x[valid_idx, :], y[valid_idx]

        clf = LogisticRegression(C=0.0001)

        clf.fit(train_x, train_y)

        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(x_test)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        print('feature len: {}'.format(len(features)))
        print('coef len: {}'.format(len(clf.coef_[0])))
        print(clf.coef_[0])
        fold_importance_df["importance"] = clf.coef_[0]
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
    # Write submission file and plot feature importance

    display_importances(feature_importance_df)
    return sub_preds, feature_importance_df


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


def main(debug=False):
    if debug == True:
        rows = 2000
    else:
        rows = None

    app_train = pd.read_csv('../input/application_train.csv', nrows=rows)
    app_test = pd.read_csv('../input/application_test.csv', nrows=rows)

    id_train = app_train.index
    id_test = app_test.index
    test_skid = app_test[['SK_ID_CURR']]

    app_all = pd.concat([app_train, app_test], axis=0)
    print('shape: {}'.format(app_all.shape))

    y = app_train['TARGET']
    ids = app_train['SK_ID_CURR']



    del app_train['SK_ID_CURR']

    missing_values = missing_values_table(app_train)
    # print(missing_values.head(40))
    # print(missing_values.tail(10))

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
    # print('train shape {}'.format(app_train.shape))
    #
    # num_cols = app_train.select_dtypes(include=[np.number]).columns.tolist()
    # cat_cols = app_train.select_dtypes(include=[np.object]).columns.tolist()
    #
    # print("\nnum_cols count:{}".format(len(num_cols)))
    # print("\ncat_cols count:{}".format(len(cat_cols)))
    #
    # print("\nnum_cols:{}".format(num_cols))
    # print("\ncat_cols:{}".format(cat_cols))
    #
    # cat_flag_cols = [col for col in cat_cols if app_train[col].nunique() == 2]
    # print('\ncat flag 特征总数：{}'.format(len(cat_flag_cols)))
    # print('cat flag cols {}'.format(cat_flag_cols))
    #
    # # 只有一个值
    # num_1_cols = [col for col in num_cols if app_train[col].nunique() == 1]
    # print('\nnum 1 特征总数：{}'.format(len(num_1_cols)))
    # print('{}'.format(num_1_cols))
    #
    # num_flag_cols = [col for col in num_cols if app_train[col].nunique() == 2]
    # print('\nnum flag 特征总数：{}'.format(len(num_flag_cols)))
    # print('num flag cols {}'.format(num_flag_cols))
    #
    # num_discreet_cols = [col for col in num_cols if 2 < app_train[col].nunique() <= 20]
    # print('\n数值离散 特征总数：{}'.format(len(num_discreet_cols)))
    # print('数值离散 {}'.format(num_discreet_cols))
    #
    # num_continuous_cols = [col for col in num_cols if 20 < app_train[col].nunique()]
    # print('\n数值连续 特征总数：{}'.format(len(num_continuous_cols)))
    # print('数值连续 {}'.format(num_continuous_cols))

    # for col in num_cols:
    #     print(app_train[col].nunique())

    # ----------------------------------------------------------------------------------------------------

    # 数据探索可视化

    # ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------
    # 离群值可视化

    # for col in num_continuous_cols:
    #     draw_feature_distribution(app_all, col)

    # ----------------------------------------------------------------------------------------------------
    #


    # fcols = 2
    # frows = len(num_continuous_cols)
    # plt.figure(figsize=(4 * fcols, 6 * frows))
    # i = 0
    #
    # for col in num_continuous_cols:
    #
    #     dat = app_train[[col, 'TARGET']].dropna()
    #
    #     i += 1
    #     plt.subplot(frows, fcols, i)
    #     sns.distplot(dat[col], fit=stats.norm)
    #     plt.title(col + ' Original')
    #     plt.xlabel('')
    #
    #     i += 1
    #     plt.subplot(frows, fcols, i)
    #     _ = stats.probplot(dat[col], plot=plt)
    #     plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[col])))
    #     plt.xlabel('')
    #     plt.ylabel('')
    #
    #
    #
    #
    #
    # plt.tight_layout(h_pad=2.5)
    # plt.savefig('./pic/num_continue.png')
    # plt.show()

    # ----------------------------------------------------------------------------------------------------

    # 编码

    # ----------------------------------------------------------------------------------------------------

    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in app_train.columns:
        if app_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(app_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(app_train[col])
                # Transform both training and testing data
                app_train[col] = le.transform(app_train[col])
                app_test[col] = le.transform(app_test[col])

                # Keep track of how many columns were label encoded
                le_count += 1
                print(col)

    print('%d columns were label encoded.' % le_count)

    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)


    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)

    # 对齐 train test

    train_labels = app_train['TARGET']

    app_train, app_test = app_train.align(app_test, join='inner', axis=1)

    # Add the target back in
    app_train['TARGET'] = train_labels

    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)

    # ----------------------------------------------------------------------------------------------------

    # 处理异常数据 DAYS_EMPLOYED_ANOM

    # ----------------------------------------------------------------------------------------------------

    app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

    # app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
    # plt.xlabel('Days Employment')
    # plt.show()

    app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
    app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    print('There are %d anomalies in the test data out of %d entries' % (
    app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

    # ----------------------------------------------------------------------------------------------------

    # baseline

    # ----------------------------------------------------------------------------------------------------

    # Drop the target from the training data
    if 'TARGET' in app_train:
        train = app_train.drop(columns=['TARGET'])
    else:
        train = app_train.copy()

    # Feature names
    features = list(train.columns)

    # Copy of the testing data
    test = app_test.copy()

    print(type(test))

    imputer = Imputer(strategy='median')
    scaler = MinMaxScaler(feature_range=(0, 1))

    imputer.fit(train)
    train = imputer.transform(train)
    test = imputer.transform(app_test)

    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print('Training data shape: ', train.shape)
    print('Testing data shape: ', test.shape)

    train = pd.DataFrame(train, columns=features)
    test = pd.DataFrame(test, columns=features)


    # ----------------------------------------------------------------------------------------------------

    # 训练模型

    # ----------------------------------------------------------------------------------------------------
    params = {'C': 0.001}
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
            'verbose': -1,}

    # oof_preds, sub_preds, feature_importance, metrics = OOFPreds(train, train_labels, test, params=params,
    #                                                              n_splits=2, clf=None)

    oof_preds, sub_preds, feature_importance, metrics = OOFPreds(train, train_labels, test, params=params_lgb, n_splits=5, clf='lgb')

    print('feature importance: {}'.format(feature_importance.shape))
    print(feature_importance.dtypes)
    display_importances(feature_importance, num_features=20, filename='lgb_20.png')

    print(metrics)

    sub_preds = pd.concat([test_skid, sub_preds], axis=1)

    sub_preds.to_csv('lgb_baseline-fillmedian-minmax-val-180627.csv', index=False)

    # ----------------------------------------------------------------------------------------------------

    # 需要处理的离群点

    # ----------------------------------------------------------------------------------------------------
    outlier = ['YEARS_BEGINEXPLUATATION_AVG', 'DAYS_EMPLOYED', 'BASEMENTAREA_AVG']



if __name__ == '__main__':
    main(debug=False)