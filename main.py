import findspark
findspark.init()
import pyspark

from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType
from pyspark.sql.types import StringType, FloatType, DoubleType
from pyspark.sql.functions import PandasUDFType, pandas_udf
import numpy as np

conf = SparkConf().setAppName("2AMD15-MS1_Part3")

sc = SparkContext(conf=conf).getOrCreate()
ss = SparkSession(sc).builder.config("spark.driver.memory", "6g").getOrCreate()

import pyspark.sql.functions as f


def main():
    weather_df = ss.read.option('header', True).csv("weather-ta2.csv")
    stocks_df = ss.read.option('header', True).csv("stocks-ta2.csv")

    weather_df.registerTempTable("df")

    mylist = ss.sql("""
    SELECT DISTINCT  u1.station_id,
        u2.station_id
    FROM df as u1 JOIN df as u2 ON u1.station_id> u2.station_id;
    """).collect()

    a = mylist[0][0]
    b = mylist[0][1]

    weather_df = weather_df.select('station_id', 'date', 'tmp_val')
    schema = weather_df.filter(weather_df.station_id == a).alias('a').join(weather_df.filter(weather_df.station_id == b)\
                                .withColumnRenamed('station_id', 'station_id_2')\
                                .withColumnRenamed('date', 'date_2')\
                                .withColumnRenamed('tmp_val', 'tmp_val_2').alias('b'),
                                 f.col("a.date") == f.col("b.date_2"), 'inner').schema

    df = ss.createDataFrame([], schema)

    for a, b in mylist:
        df = df.union(weather_df.filter(weather_df.station_id == a).alias('a').join(weather_df.filter(weather_df.station_id == b)\
                              .withColumnRenamed('station_id', 'station_id_2')\
                              .withColumnRenamed('date', 'date_2')\
                              .withColumnRenamed('tmp_val', 'tmp_val_2').alias('b'),
                               f.col("a.date") == f.col("b.date_2"), 'inner'))

    df = df.select('station_id', 'station_id_2', 'date', 'tmp_val', 'tmp_val_2')

    def max(a, b):
        if float(a) > float(b):
            return float(a)
        else:
            return float(b)

    def min(a, b):
        if float(a) > float(b):
            return float(b)
        else:
            return float(a)

    def avg(a, b):
        return (float(a) + float(b)) / 2

    max_udf = f.udf(max, FloatType())
    min_udf = f.udf(min, FloatType())
    avg_udf = f.udf(avg, FloatType())

    df = df.withColumn('min', min_udf('tmp_val', 'tmp_val_2')) \
        .withColumn('avg', avg_udf('tmp_val', 'tmp_val_2')) \
        .withColumn('max', max_udf('tmp_val', 'tmp_val_2'))

    def unique_combination(a, b):
        col = [a, b]
        col.sort()
        return '_'.join(col)

    my_udf = f.udf(unique_combination, StringType())

    df = df.withColumn('combo_id', my_udf('station_id', 'station_id_2')).select('combo_id', 'date', 'min', 'avg', 'max')

    stocks_df = stocks_df.withColumn('price', f.col('price').cast(FloatType()))

    big_df = stocks_df.alias('a').join(df.alias('b'), f.col('a.date') == f.col('b.date'), 'inner')
    big_df = big_df.withColumn("cosine_sim", f.lit(1.0).cast("float"))
    big_df = big_df.select('a.date', 'stock_name', 'combo_id', 'price', 'avg', 'min', 'max', 'cosine_sim')

    print("Big DF size: ", big_df.count())

    @pandas_udf(big_df.schema, PandasUDFType.GROUPED_MAP)
    def cos_sim(df):
        # Names of columns
        a, b = "price", "avg"  # ('min', 'max', 'avg')
        cosine_sim_col = "cosine_sim"
        df[cosine_sim_col] = float(np.dot(df[a], df[b]) / (np.linalg.norm(df[a]) * np.linalg.norm(df[b])))
        return df

    df_final = big_df.groupby(["stock_name", "combo_id"]).apply(cos_sim)
    df_cosine = df_final.groupBy(["stock_name", "combo_id"]).agg(f.avg("cosine_sim").alias("cossim")).orderBy('cossim',
                                ascending=False)

    print("𝜏 = 0.99")
    df_cosine = df_cosine.filter(df_cosine.cossim > 0.99)
    print("No. of results: ", df_cosine.count())


if __name__ == "__main__":
    main()
