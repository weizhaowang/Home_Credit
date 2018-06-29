import gc

import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
import scipy.stats as stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import collections
from lightgbm import LGBMClassifier, plot_importance
import seaborn as snss
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 200})


def rand_jitter(arr):
    nosie = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr))


def draw_feature_distribution(df, column, foldname):
    column_values = df[df[column].notna()][column]
    # group by target
    class_0_values = df[df[column].notna() & (df['TARGET']==0)][column]
    class_1_values = df[df[column].notna() & (df['TARGET']==1)][column]
    class_t_values = df[df[column].notna() & (df['TARGET'].isna())][column]
    print('\n\n', column)
    # for features with unique values >= 10
    if len(df[column].value_counts().keys()) >= 10:
        fig, ax = plt.subplots(1, figsize=(15, 4))
        if df[column].dtype == 'object':
            label_encoder = LabelEncoder()
            label_encoder.fit(column_values)
            class_0_values = label_encoder.transform(class_0_values)
            class_1_values = label_encoder.transform(class_1_values)
            class_t_values = label_encoder.transform(class_t_values)
            column_values = label_encoder.transform(column_values)
            plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, fontsize=12, rotation='vertical')

        ax.scatter(class_0_values, rand_jitter([0] * class_0_values.shape[0]), label='Class0', s=10, marker='o',
                   color='#7ac143', alpha=1)
        ax.scatter(class_1_values, rand_jitter([10] * class_1_values.shape[0]), label='Class1', s=10, marker='o',
                   color='#fd5c63', alpha=1)
        ax.scatter(class_t_values, rand_jitter([20] * class_t_values.shape[0]), label='Test', s=10, marker='o',
                   color='#037ef3', alpha=0.4)
        ax.set_title(column + ' group by target', fontsize=16)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(column + ' distribution', fontsize=16)
    else:
        all_categories = list(df[df[column].notna()][column].value_counts().keys())
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.set_title(column, fontsize=16)
        plt.xlabel('Categories', fontsize=16)
        plt.ylabel('Counts', fontsize=16)

        value_counts = class_0_values.value_counts()
        x_0 = np.arange(len(all_categories))
        y_0 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_0, y_0, color='#7ac143', width=bar_width, label='class0')

        value_counts = class_1_values.value_counts()
        x_1 = np.arange(len(all_categories))
        y_1 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_1 + bar_width, y_1, color='#fd5c63', width=bar_width, label='class1')

        value_counts = class_t_values.value_counts()
        x_2 = np.arange(len(all_categories))
        y_2 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_2 + 2 * bar_width, y_2, color='#037ef3', width=bar_width, label='test')

        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

        for i, v in enumerate(y_0):
            if y_0[i] + y_1[i] == 0:
                ax.text(i - .08, max(y_0) // 1.25, 'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i - .08, max(y_0) // 1.25, "{:0.1f}%".format(100 * y_0[i] / (y_0[i] + y_1[i])), fontsize=14,
                        rotation='vertical')

        for i, v in enumerate(y_1):
            if y_0[i] + y_1[i] == 0:
                ax.text(i - .08, max(y_0) // 1.25, 'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i + bar_width - .08, max(y_0) // 1.25, "{:0.1f}%".format(100 * y_1[i] / (y_0[i] + y_1[i])),
                        fontsize=14, rotation='vertical')

        for i, v in enumerate(y_2):
            if y_2[i] == 0:
                ax.text(i + 2 * bar_width - .08, max(y_0) // 1.25, 'Missing in Test', fontsize=14,
                        rotation='vertical')
            else:
                ax.text(i + 2 * bar_width - .08, max(y_0) // 1.25, str(y_2[i]), fontsize=14, rotation='vertical')

        plt.xticks(x_0 + 2 * bar_width / 3, all_categories, fontsize=16)

    plt.savefig('./pic/' + foldname + '/' + column +'.png')
    # plt.show()


def display_importances(feature_importance_df, num_features=20, filename=None):
    """
    feature_importance_df: dataframe 结构为 ''fold'', 'feature', 'importance'
    """
    print(feature_importance_df)
    print('display_importances')
    print(num_features, filename)
    # Plot feature importances
    cols = feature_importance_df[["feature", "importance"]].groupby(
        "feature").mean().sort_values(
            by="importance", ascending=False)[:num_features].index

    cols2 = feature_importance_df[["feature", "importance"]].groupby(
        "feature").mean().sort_values(
        by="importance", ascending=True)[:num_features].index

    best_features = feature_importance_df.loc[
        feature_importance_df.feature.isin(cols)]

    best_features2 = feature_importance_df.loc[
        feature_importance_df.feature.isin(cols2)]

    plt.figure(figsize=(16, 10))
    plt.subplot(121)
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False))

    plt.title('Features (avg over folds) max top {}'.format(num_features))

    plt.subplot(122)
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features2.sort_values(by="importance", ascending=True))


    plt.title('Features (avg over folds) minus max top {}'.format(num_features))
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def OOFPreds(X, y, test_X, params, n_splits=5, random_state=23, clf='lgb'):
    """
    输入要求数据为 Dataframe
    返回数据 Series
    """

    # 方便后续特征重要度分析
    feature_importance = pd.DataFrame(columns=['feature', 'importance', 'fold'])

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # oof 是交叉验证结果 sub是测试集预测结果
    oof_preds, sub_preds = np.zeros(X.shape[0]), np.zeros(test_X.shape[0])

    oof_train = np.zeros(X.shape[0])

    print(X.shape, test_X.shape)

    valid_scores = []
    train_scores = []

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        trn_x, trn_y = X.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]

        # 初始化 score记录方式
        trn_init_score = pd.Series([0.95] * len(trn_x), index=trn_x.index)
        val_init_score = pd.Series([0.95] * len(val_x), index=val_x.index)

        # 模型构建与预测任务
        if clf == 'lgb':
            gbm = LGBMClassifier(**params)
            gbm.fit(trn_x, trn_y, init_score=trn_init_score,
                    eval_set=[(trn_x, trn_y), (val_x, val_y)],
                    eval_init_score=[trn_init_score, val_init_score],
                    eval_metric='auc', verbose=30, early_stopping_rounds=100)

            pred_val = gbm.predict_proba(val_x, num_iteration=gbm.best_iteration_)[:, 1]
            pred_test = gbm.predict_proba(test_X, num_iteration=gbm.best_iteration_)[:, 1]

            # 预测分数 预测结果记录
            oof_preds[val_idx] = pred_val
            sub_preds += pred_test / folds.n_splits

            print(gbm.best_score_)

            valid_score = gbm.best_score_['valid_1']['auc']
            train_score = gbm.best_score_['training']['auc']

            valid_scores.append(valid_score)
            train_scores.append(train_score)

            feature_importance = feature_importance.append(pd.DataFrame({
                'importance': gbm.feature_importances_,
                'fold': [n_fold + 1] * X.shape[1],
                'feature': X.columns.tolist()}))


        else:
            # 自己的模型
            # 任务一：完成模型的构建预测任务
            # 任务二：完成预测分数，预测结果的记录
            # 任务三：完成模型重要程度的记录
            clf = LogisticRegression(**params)
            clf.fit(trn_x, trn_y)

            pred_train = clf.predict_proba(trn_x)[:, 1]
            pred_val = clf.predict_proba(val_x)[:, 1]
            pred_test = clf.predict_proba(test_X)[:, 1] \

            oof_preds[val_idx] = pred_val
            sub_preds += pred_test / folds.n_splits


            valid_score = roc_auc_score(val_y, pred_val)
            train_score = roc_auc_score(trn_y, pred_train)

            valid_scores.append(valid_score)
            train_scores.append(train_score)

            feature_importance = feature_importance.append(pd.DataFrame({
                'importance': clf.coef_[0],
                'fold': [n_fold + 1] * X.shape[1],
                'feature': X.columns.tolist()}))


        print('Fold {:02d} 训练集 AUC: {:.6f} 验证集 AUC: {:.6f} '.format(n_fold + 1, train_score, valid_score))
        del trn_x, trn_y, val_x, val_y;
        gc.collect()

    feature_importance['importance'] = feature_importance['importance'].astype(float)

    fold_names = list(range(folds.n_splits))
    fold_names.append('overall')

    valid_auc = roc_auc_score(y, oof_preds)

    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # 构建记录分数的 Dataframe
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    oof_preds = pd.Series(oof_preds.flatten(), index=X.index).rename('TARGET')
    sub_preds = pd.Series(sub_preds.flatten(), index=test_X.index).rename('TARGET')

    return oof_preds, sub_preds, feature_importance, metrics

def skew_plot(app_train, features, filename):

    fcols = 2
    frows = len(features)
    plt.figure(figsize=(4 * fcols, 6 * frows))
    i = 0
    for col in features:
        dat = app_train[[col, 'TARGET']].dropna()

        i += 1
        plt.subplot(frows, fcols, i)
        sns.distplot(dat[col], fit=stats.norm)
        plt.title(col)
        plt.xlabel('')

        i += 1
        plt.subplot(frows, fcols, i)
        _ = stats.probplot(dat[col], plot=plt)
        plt.title('skew=' + '{:.4f}'.format(stats.skew(dat[col])))
        plt.xlabel('')
        plt.ylabel('')

    plt.tight_layout(h_pad=2.5)
    plt.savefig(filename)
    plt.show()


def kde_plot(app_train, features, filename):
    plt.figure(figsize=(8, 4 * len(features)))
    for i, col in enumerate(features):
        plt.subplot(len(features), 1, i + 1)
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, col], label='target==0')
        sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, col], label='target==1')
        plt.title(col)
        plt.xlabel(col);
        plt.ylabel('Density');

    plt.tight_layout(h_pad=2.5)
    plt.savefig(filename)
    plt.show()