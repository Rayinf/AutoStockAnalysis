import re

# 读取文件
with open('data/stock.md.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找并替换stock_zh_a_hist部分
old_pattern = '''接口: stock_zh_a_hist

目标地址: https://quote.eastmoney.com/concept/sh603777.html?from=classic(示例)

描述: 东方财富-沪深京 A 股日频率数据; 历史数据按日频率更新, 当日收盘价请在收盘后获取

限量: 单次返回指定沪深京 A 股上市公司、指定周期和指定日期间的历史行情日频率数据

输入参数

| 名称         | 类型    | 描述                                                       |
|------------|-------|----------------------------------------------------------|
| symbol     | str   | symbol='603777'; 股票代码可以在 **ak.stock_zh_a_spot_em()** 中获取 |
| period     | str   | period='daily'; choice of {'daily', 'weekly', 'monthly'} |
| start_date | str   | start_date='20210301'; 开始查询的日期                           |
| end_date   | str   | end_date='20210616'; 结束查询的日期                             |
| adjust     | str   | 默认返回不复权的数据; qfq: 返回前复权后的数据; hfq: 返回后复权后的数据               |

| timeout    | float | timeout=None; 默认不设置超时参数                                  |

**股票数据复权**'''

new_content = '''接口: stock_zh_a_hist

目标地址: https://quote.eastmoney.com/concept/sh603777.html?from=classic(示例)

描述: 东方财富-沪深京 A 股日频率数据; 历史数据按日频率更新, 当日收盘价请在收盘后获取

限量: 单次返回指定沪深京 A 股上市公司、指定周期和指定日期间的历史行情日频率数据

输入参数

| 名称         | 类型    | 描述                                                       |
|------------|-------|----------------------------------------------------------|
| symbol     | str   | symbol='603777'; 股票代码可以在 **ak.stock_zh_a_spot_em()** 中获取 |
| period     | str   | period='daily'; choice of {'daily', 'weekly', 'monthly'} |
| start_date | str   | start_date='20210301'; 开始查询的日期                           |
| end_date   | str   | end_date='20210616'; 结束查询的日期                             |
| adjust     | str   | 默认返回不复权的数据; qfq: 返回前复权后的数据; hfq: 返回后复权后的数据               |

| timeout    | float | timeout=None; 默认不设置超时参数                                  |

接口示例

```python
import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20220101", end_date="20221201", adjust="")
print(stock_zh_a_hist_df)
```

数据示例

```
          日期    股票代码     开盘     收盘     最高     最低     成交量           成交额    振幅   涨跌幅   涨跌额   换手率
0  2022-01-04  000001  17.36  17.37  17.37  17.28  420920  7.305440e+08  0.52  0.00  0.00  0.43
1  2022-01-05  000001  17.30  17.20  17.30  17.10  455033  7.819340e+08  1.15 -0.98 -0.17  0.47
2  2022-01-06  000001  17.10  17.08  17.15  17.01  313951  5.360520e+08  0.82 -0.70 -0.12  0.32
3  2022-01-07  000001  17.15  17.16  17.20  17.08  280282  4.804850e+08  0.70  0.47  0.08  0.29
4  2022-01-10  000001  17.13  17.20  17.25  17.13  339356  5.833240e+08  0.70  0.23  0.04  0.35
```

**股票数据复权**'''

# 替换内容
content = content.replace(old_pattern, new_content)

# 写入文件
with open('data/stock.md.txt', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ 已为 stock_zh_a_hist 添加示例代码')

