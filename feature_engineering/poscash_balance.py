'''
为 POS_CASH_balance 生成新的特征 + 1
'''
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def main():
    poscash_bal = pd.read_csv('../input/POS_CASH_balance.csv',
                                  )  #实际1300万多条
    print(poscash_bal.shape)

    poscash_bal['sk_dpd_sub'] = poscash_bal['SK_DPD'] - poscash_bal['SK_DPD_DEF']

    print(poscash_bal.shape)
    poscash_bal.to_csv('../input/new_feature_data/POS_CASH_balance.csv')


if __name__ == '__main__':
    main()