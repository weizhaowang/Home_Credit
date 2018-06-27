'''
为 installments_payments 生成新的特征
'''
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def main():
    install_payment = pd.read_csv('../input/installments_payments.csv',
                                  )  #实际1300万多条
    print(install_payment.shape)

    # 逾期天数 （未超期为负）
    install_payment['days_overdue_sub'] = install_payment['DAYS_ENTRY_PAYMENT'] - install_payment['DAYS_INSTALMENT']
    # 没逾期的直接归零
    install_payment.loc[install_payment['days_overdue_sub'] < 0, ['days_overdue_sub']] = 0

    # 未平衡金额 = 分期金额 - 还款金额 （多还为负）
    install_payment['amt_unbalance_sub'] = install_payment['AMT_INSTALMENT'] - install_payment['AMT_PAYMENT']

    # 组合上面两个  未还金额 X 逾期时间
    install_payment['unbalance_amt_X_day_overdue'] = install_payment['days_overdue_sub'] * install_payment['amt_unbalance_sub']

    print(install_payment[install_payment['unbalance_amt_X_day_overdue'] != 0].shape)
    print(install_payment[install_payment['unbalance_amt_X_day_overdue'] != 0].head())

    install_payment.to_csv('../input/new_feature_data/installments_payments.csv')

if __name__ == '__main__':
    main()