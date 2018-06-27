'''
为 redit_card_balance 处理空值  生成特征



'''

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def main():
    credit_card_balance = pd.read_csv('../input/credit_card_balance.csv',
                                  nrows=10000)  #实际1300万多条
    print(credit_card_balance.shape)

    # print(credit_card_balance.isnull().sum())

    # ----------------------------------------------------------------------------------------------------

    # 填充空值

    # ----------------------------------------------------------------------------------------------------

    col_fillzero = ['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']

    for col in col_fillzero:
        credit_card_balance[col].fillna(0, inplace=True)

    # 最少还款 是  信用额度的 5%
    credit_card_balance.loc[credit_card_balance['AMT_INST_MIN_REGULARITY'].isnull(), 'AMT_INST_MIN_REGULARITY'] = credit_card_balance.loc[credit_card_balance['AMT_INST_MIN_REGULARITY'].isnull(),
                                                                                                           'AMT_CREDIT_LIMIT_ACTUAL'] * 0.05


    print(credit_card_balance.isnull().sum())
    # credit_card_balance['sk_dpd_sub'] = credit_card_balance['SK_DPD'] - credit_card_balance['SK_DPD_DEF']

    # print(credit_card_balance.shape)
    # credit_card_balance.to_csv('../input/new_feature_data/credit_card_balance.csv')


if __name__ == '__main__':
    main()