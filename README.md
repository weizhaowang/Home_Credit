# Home Credit Default Risk
> Kaggle 上最近的结构化数据项目，上手实践一下


## Will Koehrsen 的 Kernel 学习

Will Koehrsen 这位作者写了一系列教程，主要从 简介 -> 手工特征工程-> 利用库来进行自动特征工程，
看了两篇了觉得介绍由浅入深，值得深入学习，提升特征工程技巧，提升排名。

- 简介
[Start Here: A Gentle Introduction
](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction)
- 手工特征工程
[Introduction to Manual Feature Engineering
](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering)
- 手工特征工程2
[Introduction to Manual Feature Engineering P2](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2
)
- 自动特征工程(基础)
[Automated Feature Engineering Basics
](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics/notebook)
- 自动特征工程(探索)
[Tuning Automated Feature Engineering (Exploratory)
](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory/notebook)

### 第一篇要点

本篇作为系列开篇，只从主表的数据进行分析，特征工程也是只是挑选了四个影响力最大的四个特征，模型使用了基本的 Logistic 回归 和 随机森林。入门学习真的是非常良心。

- 工程过程
    - 空值用中位数填充 Imputer(strategy='median')
    - object 只有两个值 LabelEncode
    - object 多个值    get_dummies
    - 对四个高关联的特征构建多项式特征 PolynomialFeatures(degree=3)
    - 对数值型数据进行 MinMax 标准化缩放
    - LR + RF 模型训练

- 记录一下自己不太熟悉的一些api

`df.dtypes.value_counts()`
counts vs. value_counts  后者用的比较少，这次是分析不同特征类型的数量，记录一下

`app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0))`
select_dtypes  pd.Series.nunique 组合使用 查看每个 object 特征的 不重复值的个数

`train.align(test, join='inner', axis=1)`
在这之前提取出训练集标签，然后使用align可以过滤出测试集中没有的特征，当然重要的是保持训练测试集的维度相同

`seaborn.kdeplot`
核密度估计 用这个图来可视化 单一特征中不同值的数量分布


因为是 测试baseline，熟悉数据，所以也没有仔细调参，都直接使用作者notebook中的参数套用了

分数确实很低，不过慢慢升级的感觉更爽吧。

### 第二篇要点

**数据聚合方式**

![](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

开始利用器多个来源的数据信息。
另外的两个数据来源，左侧的两个是来自信用卡管理机构的信息，右侧的是来自本公司之前的数据。

bureau.csv 向上关联 application_train/test.csv 是根据 `SK_ID_DURR` 链接
bureau.csv 向下关联 bureau_balance.csv 是根据 `SK_ID_BUREAU` 链接

`SK_ID_DURR` -> `SK_ID_BUREAU` 是一对多的关系，也就是当前每条申请记录在bureau.csv 中可能有多个账户
`SK_ID_BUREAU` 中每个账户也每个月都有信息
所以，将新表融合进我们主表的过程，是从下向上不断 `分组 -> 聚合 -> 链接` 的过程
![](https://ws2.sinaimg.cn/large/006tKfTcgy1fshmi5yejuj30qf0ad3zk.jpg)

对应左边本公司数据也是一样。

**针对不同类型数据进行不同的聚合**

```
客户之前的账户总数 -> 计数信息
数值型 -> min max sum median
分类型 -> 各分类的个数 占比
```
- 数值型实例
![](https://ws1.sinaimg.cn/large/006tKfTcgy1fshpobksesj30k205eq3a.jpg)

- 分类型实例

![](https://ws1.sinaimg.cn/large/006tKfTcgy1fshppq0gnqj30a80ks0tg.jpg)
![](https://ws2.sinaimg.cn/large/006tKfTcgy1fshpqzyl5xj31b408sdgr.jpg)

**代码操作**

```python
def agg_numeric(df, group_var, df_name):
    counts = pd.DataFrame(df.groupby(group_var, as_index = False)[df.columns[1]].count()).rename(columns = {df.columns[1]: '%s_counts' % df_name})

    # Group by the specified variable and calculate the statistics
    agg = df.groupby(group_var).agg(['mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    #  Rename the columns
    agg.columns = columns

    # Merge with the counts
    agg = agg.merge(counts, on = group_var, how = 'left')

    return agg
```

```python
def count_categorical(df, group_var, df_name):
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])

    column_names = []

    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))

    categorical.columns = column_names

    return categorical
```

### 第三篇要点
Automated Feature Engineering Basics

自动特征工程基础

#### 3.1 特征深度综合

看到自动特征工程这种描述，脑子里总是忍不住幻想，有个开源库，我把数据扔进去，数据直接填充好，清洗干净，生成了足够的特征，而且做过了筛选。。。。。。

当然没这种好事，featuretool这个库主要使用场景是数据源的表比较多，当然就是这次竞赛的场景。作用主要在自动化的进行**特征深度综合**。

![](https://ws3.sinaimg.cn/large/006tNc79gy1fsiukivet2j315p0ex412.jpg)
我们在之前已经实现了纯手工`分组 -> 聚合 -> 链接`过程，这是一张官方blog给出的介绍用图，这种聚合过程可以看做是浅层的特征聚合。

![](https://ws4.sinaimg.cn/large/006tNc79gy1fsiv7peuvxj30sa0b7q4n.jpg)

当表的关系比较复杂，有多层id关系的时候，例如在网购场景中 `用户id -> 订单id -> 商品id` 这种每层都是一对多的表关系。图中生成的特征含义是：*每位客户，订单中所有商品总价的均值*

因为每层的聚合关系都有多种，导致了生成特征的数量随着聚合层数成指数级增长。

当然特征工程的一条路径是根据我们对于数据的先验知识进行特征工程，但是再丰富的知识也会存在遗漏和盲区，所以featuretool的工作可以作为生成特征的有效补充。


以下是官方的博客简介和最早分享的论文
- [Deep Feature Synthesis: How Automated Feature Engineering Works](https://www.featurelabs.com/blog/deep-feature-synthesis/)
- [Deep Feature Synthesis:Towards Automating Data Science Endeavors](https://www.featurelabs.com/wp-content/uploads/2017/12/DSAA_DSM_2015-1.pdf)


#### 3.2 featuretool 基本概念

- **实体** 和 **实体集合** Entities and EntitySets
- **表关系** Relationships between tables
- **特征基本操作** Feature primitives

**实体**
实体简单的认为就是一张表或者是pandas中的dataframe

**实体集合**
实体集合是一系列的表和他们组成关系的集合

构建实体集合 和 在实体集合中添加实体的基本操作
```python
es = ft.EntitySet(id="transactions")
es = es.entity_from_dataframe(entity_id="transactions",
   ...:                               dataframe=transactions_df,
   ...:                               index="transaction_id",
   ...:                               time_index="transaction_time",
   ...:                               variable_types={"product_id": ft.variable_types.Categorical})
```

**表关系**
表关系的概念也存在于各种关系型数据库之中，对于一对多最好的类比就是 `父母(parent) -> 孩子(child)`
注意不要有菱形图关系，也就是一个孩子存在多个父母的关系，以防止聚合时候的冲突。

构建表关系 在实体集合中添加表关系的基本操作
```python
new_relationship = ft.Relationship(es["products"]["product_id"],
   ...:                                    es["transactions"]["product_id"])

es = es.add_relationship(new_relationship)   
```


**特征基本操作**

特征剧本操作有两个大类
- 统计聚合 Aggregation
- 变换 Transformation

常用的操作：
![](https://ws3.sinaimg.cn/large/006tNc79gy1fsiwlzv2ipj30ip09r75g.jpg)

![](https://ws4.sinaimg.cn/large/006tNc79gy1fsiwmgmjqdj30mi09kt9z.jpg)

当然也可以自定义**特征基本操作**，在第四篇中应该涉及到了

基本操作
```
feature_matrix, feature_defs = ft.dfs(entityset=es,
   ...:                                       target_entity="customers",
   ...:                                       agg_primitives=["count"],
   ...:                                       trans_primitives=["month"],
   ...:                                       max_depth=1)
```
将需要使用的基本操作，作为参数传入dfs方法，获取特征矩阵。

[完整的featuretool代码 为Home Credit Default Risk问题进行自动特征工程](https://www.kaggle.com/willkoehrsen/feature-engineering-using-feature-tools)
```python
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# Clearing up memory
import gc

# Featuretools for automated feature engineering
import featuretools as ft

# Suppress pandas warnings
import warnings
warnings.filterwarnings('ignore')

print('Reading in data')

# Read in the full datasets
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
cash = pd.read_csv('../input/POS_CASH_balance.csv')
credit = pd.read_csv('../input/credit_card_balance.csv')
previous = pd.read_csv('../input/previous_application.csv')
installments = pd.read_csv('../input/installments_payments.csv')

# Join the application dataframes together
app_test['set'] = 'test'
app_test['TARGET'] = -999
app_train['set'] = 'train'

# Append the dataframes (this is a row bind in R)
app = app_train.append(app_test, ignore_index = True)

# Create the entity set with an id
es = ft.EntitySet(id = 'applications')

# Add in all the entities

# Entities with a unique index
es = es.entity_from_dataframe(entity_id = 'app', dataframe = app, index = 'SK_ID_CURR')

es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')

es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, index = 'SK_ID_PREV')

# Entities that do not have a unique index
es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe = bureau_balance,
                              make_index = True, index = 'bb_index')

es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash,
                              make_index = True, index = 'cash_index')

es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments,
                              make_index = True, index = 'in_index')

es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,
                              make_index = True, index = 'credit_index')


# Relationship between app and bureau
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

# Relationship between current app and previous apps
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])

print(es)

print('Clearing up memory')

gc.enable()
# Clear up memory
del app, bureau, bureau_balance, cash, credit, installments, previous_app
gc.collect()

print('Deep Feature Synthesis in Progress')

# Default primitives from featuretools
default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

# DFS for application features using a max depth of 2
feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'app',
                       trans_primitives = default_trans_primitives,
                       agg_primitives=default_agg_primitives,
                       max_depth = 2, features_only=False, verbose = True)

# Reset the index to make SK_ID_CURR a column again                                      
feature_matrix = feature_matrix.reset_index()

print('Saving features')
feature_matrix.to_csv('feature_matrix.csv', index = False)
```

当前问题

### 第四篇要点




## 自己看特征含义的收获

#### 主表相关

`REG_REGION_NOT_LIVE_REGION	REG_REGION_NOT_WORK_REGION	LIVE_REGION_NOT_WORK_REGION	REG_CITY_NOT_LIVE_CITY	REG_CITY_NOT_WORK_CITY	LIVE_CITY_NOT_WORK_CITY`

各种地址之间的关系  可以计算一下总和

各种提供证件  可以做个求和

各个时间段的查询次数可以做个 求和？ 或者 给不同时间段做个不同的权重？

#### previous_application

`DAYS_FIRST_DRAWING	DAYS_FIRST_DUE	DAYS_LAST_DUE_1ST_VERSION	DAYS_LAST_DUE`  +  `CNT_PAYMENT` + `AMT_CREDIT`

到期时间 + 贷款分期期数 + 最终审批金额   是不是可以组合一下  

想要生成一个  还有多少贷款没还的特征  当然越多没还越容易违约

在和current 的 application 做个比率  

`AMT_APPLICATION	AMT_CREDIT`

申请金额  和  审批金额是不是可以做个 比率特征  审批的比率越高应该越容易还贷吧？？？

`AMT_APPLICATION	AMT_CREDIT` + `AMT_GOODS_PRICE`
贷款金额  和  商品金额可以做个 比率   应该是比率越小越容易还贷


- credit_card_balance
`AMT_DRAWINGS_ATM_CURRENT	AMT_DRAWINGS_CURRENT	AMT_DRAWINGS_OTHER_CURRENT	AMT_DRAWINGS_POS_CURRENT`

生成占比特征

#### POS_CASH_balance

逾期天数 （days_overdue_sub 未超期为负 将负数归零）= 分期到期时间(DAYS_ENTRY_PAYMENT)

未平衡金额(amt_unbalance_sub 多还为负) = 分期金额(AMT_INSTALMENT) - 还款金额(AMT_PAYMENT)

组合上面两个  未还金额 X 逾期时间



#### bureau 相关

`CREDIT_DAY_OVERDUE`  一般 0 和 负无穷 是良好状态， 给0赋值 min？？

`AMT_CREDIT_SUM_LIMIT`  理论上应该等于 `AMT_CREDIT_SUM` - `AMT_CREDIT_SUM_DEBT`

但是实际上很乱，只有一部分符合

据观察 有的人在信用卡里存钱。。。。没有额度，负债还是复制  有点尴尬。。。

负债超过信用额度的  可以新建一个特征

`AMT_BALANCE	AMT_CREDIT_LIMIT_ACTUAL` 平衡金额 大于 信用金额  说明 边消费边还款啊 可以计算差值
