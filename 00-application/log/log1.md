

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
