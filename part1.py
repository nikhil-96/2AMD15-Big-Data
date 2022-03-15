# import findspark
# findspark.init()

from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format
from pyspark.sql.types import StructType, IntegerType
from pyspark.sql.types import StringType, FloatType, DoubleType, ArrayType
# from pyspark.sql.functions import PandasUDFType, pandas_udf
import numpy as np
from pyspark.sql import Window
import sys
import datetime
import pandas as pd

conf = SparkConf().setAppName("2AMD15-MS1_Part1").setMaster("local[*]")

# sc = SparkContext(conf=conf).getOrCreate()
ss = SparkSession.builder.config(conf=conf).appName("Milestone 1 - Group 29").getOrCreate()

import pyspark.sql.functions as f

def main():
    df = ss.read.csv('/MS1.txt')
    newdf = df.select(df._c0.alias('stock_name'),
                      date_format(f.to_timestamp('_c1', 'mm/dd/yyyy'), "yyyy-mm-dd").alias('date'),
                      df._c2.alias('price'),
                      df._c3.alias('volume')).filter((f.year('date') >= 2016) & (f.year('date') <= 2020)) \
        .dropDuplicates(['stock_name', 'date'])

    dates = [row.date for row in newdf.select('date').distinct().collect()]
    dates.sort()
    dates2 = dates[:1000]

    stocks = [row.stock_name for row in
              newdf.where(newdf.date.isin(dates2)).groupBy('stock_name').count().orderBy('count', ascending=False).take(
                  1000)]
    chosen_stocks = stocks[:1000]

    df_small = newdf.filter(newdf.stock_name.isin(chosen_stocks)) \
        .filter((dates2[0] <= newdf.date) & (newdf.date <= dates2[-1]))
    df_small = df_small.cache()
    df.unpersist()
    newdf.unpersist()

    df_small = df_small.withColumn('date', f.col('date')) \
        .withColumn("date_existent", f.col("date"))

    def func():
        return dates2

    func_udf = f.udf(func, ArrayType(StringType()))

    df_base = df_small.select('stock_name').distinct().withColumn('date', f.explode(func_udf())) \
        .withColumn('date', f.col('date'))

    df_full = df_base.join(df_small, ['stock_name', 'date'], "leftouter")

    df_small.unpersist()
    df_full = df_full.cache()

    window_ff = Window.partitionBy('stock_name') \
        .orderBy('date') \
        .rowsBetween(-sys.maxsize, 0)

    window_bf = Window.partitionBy('stock_name') \
        .orderBy('date') \
        .rowsBetween(0, sys.maxsize)

    read_last = f.last(df_full['price'], ignorenulls=True).over(window_ff)
    readdate_last = f.last(df_full['date_existent'], ignorenulls=True).over(window_ff)

    read_next = f.first(df_full['price'], ignorenulls=True).over(window_bf)
    readdate_next = f.first(df_full['date_existent'], ignorenulls=True).over(window_bf)

    df_filled = df_full.withColumn('price_ff', read_last) \
        .withColumn('date_ff', readdate_last) \
        .withColumn('price_bf', read_next) \
        .withColumn('date_bf', readdate_next)

    def interpol(x, x_prev, x_next, y_prev, y_next, y):
        if x_prev == x_next:
            return float(y)
        else:
            # if (y_prev is not None) & (y_next is not None):
            #    m = (float(y_next)-float(y_prev))/(datetime.date(int(x_next[:3]), int(x_next[5:7]), int(x_next[-2:])) - datetime.date(int(x_prev[:3]), int(x_prev[5:7]), int(x_prev[-2:]))).days
            #    y_interpol = float(y_prev) + m * (datetime.date(int(x[:3]), int(x[5:7]), int(x[-2:])) - datetime.date(int(x_prev[:3]), int(x_prev[5:7]), int(x_prev[-2:]))).days
            if (y_prev is not None) & (y_next is not None):
                m = (float(y_next) - float(y_prev)) / (
                            datetime.datetime.strptime(x_next, '%Y-%m-%d') - datetime.datetime.strptime(x_prev,
                                                                                                        '%Y-%m-%d')).days
                y_interpol = float(y_prev) + m * (
                            datetime.datetime.strptime(x, '%Y-%m-%d') - datetime.datetime.strptime(x_prev,
                                                                                                   '%Y-%m-%d')).days
            if x_prev is not None:
                y_interpol = y_prev
            else:
                y_interpol = y_next
            return float(y_interpol)

    interpol_udf = f.udf(interpol, FloatType())
    df_filled = df_filled.withColumn('price_interpol',
                                     interpol_udf('date', 'date_ff', 'date_bf', 'price_ff', 'price_bf', 'price')) \
        .drop('date_existent', 'date_ff', 'date_bf', 'price', 'value', 'price_bf', 'price_ff') \
        .withColumnRenamed('price_interpol', 'price')
    df_filled.repartition(1000)

    df_filled.repartition(1).toPandas().to_csv('/stocks-ta2.csv')

    files_list = ['/out_2016.txt',
                  '/out_2017.txt',
                  '/out_2018.txt',
                  '/out_2019.txt',
                  '/out_2020.txt']
    df2 = ss.read.csv(files_list, header=False)
    df2 = df2.filter(df2._c0 != 'STATION_ID')

    split_col = f.split(df2['_c3'], ' ')

    df2 = df2.withColumn('wind_angle', split_col.getItem(0))
    df2 = df2.withColumn('wind_qa', split_col.getItem(1))
    df2 = df2.withColumn('wind_type', split_col.getItem(2))
    df2 = df2.withColumn('wind_speed', split_col.getItem(3))
    df2 = df2.withColumn('wind_qs', split_col.getItem(4))

    split_col = f.split(df2['_c4'], ' ')

    df2 = df2.withColumn('tmp_val', split_col.getItem(0))
    df2 = df2.withColumn('tmp_q', split_col.getItem(1))

    split_col = f.split(df2['_c5'], ' ')

    df2 = df2.withColumn('dew_val', split_col.getItem(0))
    df2 = df2.withColumn('dew_q', split_col.getItem(1))

    split_col = f.split(df2['_c6'], ' ')

    df2 = df2.withColumn('slp_val', split_col.getItem(0))
    df2 = df2.withColumn('slp_q', split_col.getItem(1))

    df2 = df2.drop('_c2', '_c3', '_c4', '_c5', '_c6', '_c7', '_c8', '_c9', '_c10', '_c11', '_c12', '_c13')

    df2 = df2.withColumnRenamed('_c0', 'station_id').withColumnRenamed('_c1', 'date')
    df2 = df2.dropDuplicates(['station_id', 'date'])
    df2 = df2.withColumn("dew_val", df2.dew_val.cast('int'))
    df2 = df2.withColumn("tmp_val", df2.tmp_val.cast('int'))
    dew_vals = [row.dew_val for row in df2.select('dew_val').distinct().collect()]
    tmp_vals = [row.tmp_val for row in df2.select('tmp_val').distinct().collect()]
    dew_vals.sort()
    tmp_vals.sort()
    df3 = df2.select(['station_id', 'date', 'tmp_val', 'dew_val']).filter(
        (df2.tmp_val != 9999) & (df2.dew_val != 99999) & (df2.dew_val != 9999))
    df3.cache()
    df2.unpersist()

    dates = func()
    dates.sort()
    stations = [row.station_id for row in df3.filter(df3.date.isin(list(dates))).groupBy('station_id') \
        .count().orderBy('count', ascending=False) \
        .take(1000)]

    dates = list(dates)
    dates.sort()
    df_small = df3.filter(df3.station_id.isin(stations)) \
        .filter((dates[0] <= df3.date) & (df3.date <= dates[-1]))

    df_small = df_small.cache()
    df3.unpersist()

    df_small = df_small.withColumn('date', f.col('date')) \
        .withColumn("date_existent", f.col("date"))

    func_udf = f.udf(func, ArrayType(StringType()))
    df_base = df_small.select('station_id').distinct().withColumn('date', f.explode(func_udf())) \
        .withColumn('date', f.col('date'))

    df_full = df_base.join(df_small, ['station_id', 'date'], "leftouter")

    df_small.unpersist()
    df_full = df_full.cache()
    window_ff = Window.partitionBy('station_id') \
        .orderBy('date') \
        .rowsBetween(-sys.maxsize, 0)

    window_bf = Window.partitionBy('station_id') \
        .orderBy('date') \
        .rowsBetween(0, sys.maxsize)

    date_last = f.last(df_full['date_existent'], ignorenulls=True).over(window_ff)
    date_next = f.first(df_full['date_existent'], ignorenulls=True).over(window_bf)

    tmp_last = f.last(df_full['tmp_val'], ignorenulls=True).over(window_ff)
    tmp_next = f.first(df_full['tmp_val'], ignorenulls=True).over(window_bf)

    dew_last = f.last(df_full['dew_val'], ignorenulls=True).over(window_ff)
    dew_next = f.first(df_full['dew_val'], ignorenulls=True).over(window_bf)

    df_filled = df_full.withColumn('dew_ff', dew_last) \
        .withColumn('dew_bf', dew_next) \
        .withColumn('tmp_ff', tmp_last) \
        .withColumn('tmp_bf', tmp_next) \
        .withColumn('date_ff', date_last) \
        .withColumn('date_bf', date_next)

    interpol_udf = f.udf(interpol, IntegerType())
    df_filled = df_filled.withColumn('tmp_interpol',
                                     interpol_udf('date', 'date_ff', 'date_bf', 'tmp_ff', 'tmp_bf', 'tmp_val')) \
        .withColumn('dew_interpol',
                    interpol_udf('date', 'date_ff', 'date_bf', 'dew_ff', 'dew_bf', 'dew_val')) \
        .drop('date_existent', 'date_ff', 'date_bf',
              'dew_val', 'dew_bf', 'dew_ff',
              'tmp_val', 'tmp_bf', 'tmp_ff') \
        .withColumnRenamed('dew_interpol', 'dew_val') \
        .withColumnRenamed('tmp_interpol', 'tmp_val')
    df_filled.repartition(1000)
    df_filled.repartition(1).toPandas().to_csv('/weather-ta2.csv')

if __name__ == "__main__":
    main()


