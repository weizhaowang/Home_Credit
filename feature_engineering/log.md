## app

家庭类型 家庭成员个数 可能存在共线性




## install_payment.py

新生成 3个特征  

这是第三个的  有值的情况  大概只有 5% 有值 其他都是空值  
(13605401, 8)
(656367, 11)

## poscash_balance.py

只是加了一列 超期差值的特征

## credit_card_balance.py

AMT_BALANCE                     
AMT_CREDIT_LIMIT_ACTUAL
两列没有空值 后续填充需要参考这两列

----

三个消费渠道直接填充 0

AMT_DRAWINGS_ATM_CURRENT         
AMT_DRAWINGS_CURRENT             
AMT_DRAWINGS_OTHER_CURRENT       
AMT_DRAWINGS_POS_CURRENT  
       
如果没有从这个渠道取钱就是 0 ，含义填充，
填充0
----
最少还贷 是 信用额度的 5%
AMT_INST_MIN_REGULARITY

