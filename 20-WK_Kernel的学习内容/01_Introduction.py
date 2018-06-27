import pandas as pd
import numpy as np

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, Imputer, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')


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


def main():
    app_train = pd.read_csv('../input/application_train.csv')  # 训练数据 (307511, 122) 可以先少读点数据跑通逻辑
    app_test = pd.read_csv('../input/application_test.csv')    #  (48744, 121)
    print(app_train.shape, app_test.shape)

    # 检查空值情况
    # missing_train = missing_values_table(app_train)
    # missing_test = missing_values_table(app_test)
    # print(missing_train.head(20))
    # print(missing_test.head(20))

    # 统计特征类型，使用不同方式处理
    print("\n不同特征类型计数 \n{}".format(app_test.dtypes.value_counts()))
    print('\n object 类型特征的不重复值数量 \n{}'.format(app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)))

    # ----------------------------------------------------------------------------------------------------

    # 对分类数据进行编码

    # ----------------------------------------------------------------------------------------------------

    # LabelEncoder -> object(2)

    le = LabelEncoder()
    le_count = 0

    for col in app_train:
        if app_train[col].dtype == 'object':
            if len(list(app_train[col].unique())) <= 2:
                le.fit(app_train[col])
                app_train[col] = le.transform(app_train[col])
                app_test[col] = le.transform(app_test[col])
                print(col)
                le_count += 1
    print('{} 列被LabelEncode编码'.format(le_count))

    # get_dummies() -> object(>2)

    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)
    # print(app_train.shape, app_test.shape)

    # align 对齐 训练集 测试集

    train_labels = app_train['TARGET']
    app_train, app_test = app_train.align(app_test, join='inner', axis=1)
    # print(app_train.shape, app_test.shape)

    # ----------------------------------------------------------------------------------------------------

    # 探索 与label 相关性强的 4 个特征

    # ----------------------------------------------------------------------------------------------------

    # DAYS_BIRTH -> YEARS_BIRTH

    app_train['TARGET'] = train_labels
    correlations = app_train.corr()['TARGET'].sort_values()

    print("\n正相关靠前")
    print(correlations.tail(15))
    print('\n负相关靠前')
    print(correlations.head(15))

    app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
    print(app_train['DAYS_BIRTH'].corr(app_train['TARGET']))

    plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
    plt.title('Age of Client');
    plt.xlabel('Age (years)');
    plt.ylabel('Count')
    # plt.show()

    plt.figure(figsize=(10, 8))
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH']/365,
                label='target == 0')
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH']/365,
                label='target == 1')

    # plt.show()
    age_data = app_train[['TARGET', 'DAYS_BIRTH']]
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH']/365
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))

    age_groups = age_data.groupby('YEARS_BINNED').mean()
    print(age_groups)

    #  'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'  附带上 一起看看 days_birth

    ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    ext_data_corrs = ext_data.corr()
    print(ext_data_corrs)

    sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
    plt.title('Correlation Heatmap')
    # plt.show()

    plt.figure(figsize=(10, 12))

    # iterate through the sources
    for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):

        plt.subplot(3, 1, i + 1)

        sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label='target == 0')
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label='target == 1')

        plt.title('Distribution of %s by Target Value' % source)
        plt.xlabel('%s' % source);
        plt.ylabel('Density');

    plt.tight_layout(h_pad=2.5)
    # plt.show()

    # pair plot

    # plot_data = ext_data.drop(columns=['DAYS_BIRTH']).copy()
    # plot_data = plot_data.dropna().loc[:10000, :]
    #
    # def corr_func(x, y, **kwargs):
    #     r = np.corrcoef(x, y)[0][1]
    #     ax = plt.gca()
    #     ax.annotate("r = {:.2f}".format(r),
    #                 xy=(.2, .8), xycoords=ax.transAxes,
    #                 size=20)
    #
    # grid = sns.PairGrid(data=plot_data, size=3, diag_sharey=False,
    #                     hue='TARGET',
    #                     vars=[x for x in list(plot_data.columns) if x != 'TARGET'])
    #
    # grid.map_upper(plt.scatter, alpha=0.2)
    # grid.map_diag(sns.kdeplot)
    # grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r)
    #
    # plt.suptitle('Ext Source and Age Features Pairs Plot', size=32, y=1.05)
    # plt.show()

    # ----------------------------------------------------------------------------------------------------

    # 特征工程 多项式特征工程

    # ----------------------------------------------------------------------------------------------------

    imputer = Imputer(strategy='median')

    poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
    poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

    poly_target = app_train['TARGET']
    poly_features = poly_features.drop(columns=['TARGET'])

    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)

    poly_transformer = PolynomialFeatures(degree=3)

    poly_transformer.fit(poly_features)

    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)

    print('多项式特征训练集 shape {}'.format(poly_features.shape))

    print(
    poly_transformer.get_feature_names(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                       'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]
    )

    poly_features = pd.DataFrame(poly_features,
                                 columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                             'EXT_SOURCE_3', 'DAYS_BIRTH']))

    poly_features['TARGET'] = poly_target

    poly_corrs = poly_features.corr()['TARGET'].sort_values()

    print(poly_corrs.head(10))
    print(poly_corrs.tail(5))

    # ----------------------------------------------------------------------------------------------------
    # 数据集  + 新的特征
    poly_features_test = pd.DataFrame(poly_features_test,
                                      columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                  'EXT_SOURCE_3', 'DAYS_BIRTH']))

    # Merge polynomial features into training dataframe
    poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
    app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')

    # Merge polnomial features into testing dataframe
    poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
    app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

    # Align the dataframes
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

    # Print out the new shapes
    print('Training data with polynomial features shape: ', app_train_poly.shape)
    print('Testing data with polynomial features shape:  ', app_test_poly.shape)

    # ----------------------------------------------------------------------------------------------------

    # 准备训练测试数据 1 填充空值 Imputer  2 标准化 MinMaxScaler

    # ----------------------------------------------------------------------------------------------------

    if 'TARGET' in app_train:
        train = app_train.drop(columns=['TARGET'])
    else:
        train = app_train.copy()

    features = list(train.columns)

    test = app_test.copy()
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

    # ----------------------------------------------------------------------------------------------------

    # 构建模型训练数据，预测结果

    # ----------------------------------------------------------------------------------------------------

    log_reg = LogisticRegression(C=0.0001)

    log_reg.fit(train, train_labels)
    log_reg_pred = log_reg.predict_proba(test)[:, 1]
    submit = app_test[['SK_ID_CURR']]
    submit['TARGET'] = log_reg_pred

    submit.to_csv('log_reg_baseline.csv', index=False)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=50)
    random_forest.fit(train, train_labels)

    predictions = random_forest.predict_proba(test)[:, 1]
    submit = app_test[['SK_ID_CURR']]
    submit['TARGET'] = predictions

    submit.to_csv('random_forest_baseline.csv', index=False)


if __name__ == '__main__':
    main()