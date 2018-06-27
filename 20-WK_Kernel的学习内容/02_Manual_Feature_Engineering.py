import warnings

import pandas as pd
import numpy as np
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (agg_numeric, count_categorical,
                   kde_target, target_corrs)

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


def main():
    bureau = pd.read_csv('../input/bureau.csv')
    bureau_balance = pd.read_csv('../input/bureau_balance.csv')

    previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
        columns={'SK_ID_BUREAU': 'previous_loan_counts'})

    train = pd.read_csv('../input/application_train.csv', nrows=10000)
    train = train.merge(previous_loan_counts, on='SK_ID_CURR', how='left')

    print("新特征 previous_loan_counts 数量 {} / {}".format(train.isnull().sum()['previous_loan_counts'], train.shape[0]))

    # 总计有 44020 个空值，也就是，彻底的第一次信贷申请
    train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)

    kde_target('previous_loan_counts', train)

    bureau_agg = agg_numeric(bureau, group_var='SK_ID_CURR', df_name='burea')
    bureau_counts = count_categorical(bureau, group_var='SK_ID_CURR', df_name='bureau')

    bureau_balance_counts = count_categorical(bureau_balance, group_var='SK_ID_BUREAU', df_name='bureau_balance')
    bureau_balance_agg = agg_numeric(bureau_balance, group_var='SK_ID_BUREAU', df_name='bureau_balance')


if __name__ == '__main__':
    main()