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

### 第一篇的实践和笔记

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

### 




