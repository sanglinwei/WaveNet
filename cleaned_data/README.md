
## data processing
1. bin
2. one hot

## considering weather information

## 数据说明
- Austin中的文件里是包括一月到六月的清洗之后的数据，按每个月分别存储至不同csv；
    -  数据的第一列是时间(local time)；
    - 数据的最后一列是(average)；
    - 其他列是不同用户的功率序列；
- Boulder中的文件类似
- accumulate中的文件是聚合所有月份数据到一个csv文件中；
- metadata.csv 是对dataid 的描述，包括所在城市，房屋设施等；
- image 中文件是同一用户在一个月内的负荷变化, 具有较强规律性
- 单位：kW