---
layout: post
title: Pandas Notes
description: 一个开源的python类库
categories: Notes
tags: [Notes]
---

# Pandas数据分析

## 前言：

一个开源的python类库，数据分析、数据处理、数据可视化

虽然用python本身可以实现，但是用Pandas可以更加高性能的实现。

比自己写for循环快很多，可以跟其他类库一块使用，numpy数学计算，scikit_learn机器学习。很好的配合完成数据分析和机器学习。



anaconda中已经安装好了，几乎所有机器学习所需要的类库。也可以解决环境问题。

jupyter交互性，探索性，适合反复回头修改，看每一步运行的结果

pycharm大而全的集成开发环境，适合复杂项目的开发



## 读取数据——将其他类型的文件读取成pandas数据结构

读取表格类型的，二维的有行有列的读取。读取成pandas的对象

csv逗号分割，tsv用\t分割，txt文本分隔符随意

```python
a = pd.read_csv(path)

pd.read_txt(path)

a = pd.read_excel(path)

a = pd.read_sql("select * from 表名",con=conn)#数据库的连接，connection
```

pymysql.connect类库



### csv文件逗号分割

```python
a.head()#查看数据前几行

a.shape#查看数据的形状、返回行数、列数

a.columns#查看列名列表

a.index#查看索引列
```



## Pandas数据结构

——DataFrame、Series



为什么要区分出来一个Series，因为一维的可以抽象成一个字典，处理起来比二维的dataframe更加便捷。

DataFrame、Series二者均有索引。



### Series的生成：

#### 1通过转换列表得到

创建以数字为索引的Series（默认）



创建自定义索引的Series



#### 2通过字典生成



### Series的读取：



## Pandas查询数据——df为打开.csv文件后创建的对象

## .loc和.iloc的区别是什么？

.loc和.iloc都是pandas工具中定位某一行的函数，其中loc是location的意思，而iloc中的 i 指的是Integer。二者的区别如下：

- loc：通过行标签名称（tianqi）索引行数据。
- iloc：通过行号（0，1，2，3）索引行数据。



**查询的时候会出现数据降维的情况：**

查询的是dataFrame返回的是Series的数据，查询的Series返回的是具体的数值，查询值返回的就是值了

索引为默认的从0开始计算的

数据：beijing_tianqi_2018.csv

把日期当作普通的字符串来处理



### .loc单标签查询



### 数值区间范围查询：.loc[行，列]

传入行的区间和某一列

传入列的区间和某一行

传入行和列的区间



### 条件表达式查询，类似数据的查询语句，编写查询语句：.loc[行，列]

.loc[表达式1（行），表达式2（列）]

返回条件表达式中返回结果为True的数值。



**自己编写函数，并且调用。自己定查询规则。**
