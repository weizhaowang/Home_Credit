

# logsitc baseline

- 编码
- 处理DAYS_EMPLOYED 365243
- fill median
- maxmin scaler

```
      fold     train     valid
0        0  0.731619  0.733108
1        1  0.734051  0.724274
2        2  0.732281  0.731479
3        3  0.731528  0.734834
4        4  0.732080  0.732027
5  overall  0.732312  0.731140
```
与之前做过 特征多项式的 baseline 相比 增加了异常值处理 和 scaler 性能提升了很多

0.671 -> 0.718  + 0.045


## lgb baseline

```
      fold     train     valid
0        0  0.817856  0.761383
1        1  0.834124  0.758432
2        2  0.822590  0.759136
3        3  0.814665  0.763950
4        4  0.816481  0.763113
5  overall  0.821143  0.761175
```
- 编码
- 处理DAYS_EMPLOYED 365243
- fill median
- maxmin scaler

train 比 valid 高 有点过拟合

[[Updated 0.792 LB] LightGBM with Simple Features
](https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code)

LB 0.750 没有进行特征工程，没有结合其他table，模型已经到0.75了。。。。 真是模型的力量，或者是参数给定的好。。。
0.718  -> 0.075   + 0.032 

