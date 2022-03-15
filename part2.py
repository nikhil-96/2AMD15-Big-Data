# import findspark
# findspark.init()

from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType
from pyspark.sql.types import StringType, FloatType, DoubleType
# from pyspark.sql.functions import PandasUDFType, pandas_udf
import numpy as np

conf = SparkConf().setAppName("2AMD15-MS1_Part2").setMaster("local[*]")

# sc = SparkContext(conf=conf).getOrCreate()
ss = SparkSession.builder.config(conf=conf).appName("Milestone 1 - Group 29").getOrCreate()

import pyspark.sql.functions as f


def main():
    print("********* MS1 - part 1 ************")
    weather_df = ss.read.option('header', True).csv("/weather-ta2.csv").limit(20000).repartition('station_id')
    stocks_df = ss.read.option('header', True).csv("/stocks-ta2.csv").limit(20000).repartition('date')

    print("********* MS1 - part 2 ************")

    weather_df = weather_df.select('station_id', 'date', 'tmp_val')
    df = weather_df.alias('a').join(weather_df.withColumnRenamed('station_id', 'station_id_2') \
                                    .withColumnRenamed('date', 'date_2') \
                                    .withColumnRenamed('tmp_val', 'tmp_val_2').alias('b'),
                                    f.col("a.date") == f.col("b.date_2"), 'inner').filter('station_id > station_id_2')

    df = df.select('station_id', 'station_id_2', 'date', 'tmp_val', 'tmp_val_2').cache()
    weather_df.unpersist()

    def max_x(a, b):
        if float(a) > float(b):
            return float(a)
        else:
            return float(b)

    def min_x(a, b):
        if float(a) > float(b):
            return float(b)
        else:
            return float(a)

    def avg(a, b):
        return (float(a) + float(b)) / 2

    max_udf = f.udf(max_x, FloatType())
    min_udf = f.udf(min_x, FloatType())
    avg_udf = f.udf(avg, FloatType())

    df = df.withColumn('min', min_udf('tmp_val', 'tmp_val_2')) \
        .withColumn('avg', avg_udf('tmp_val', 'tmp_val_2')) \
        .withColumn('max', max_udf('tmp_val', 'tmp_val_2'))

    def unique_combination(a, b):
        col = [a, b]
        col.sort()
        return '_'.join(col)

    my_udf = f.udf(unique_combination, StringType())

    df = df.withColumn('combo_id', my_udf('station_id', 'station_id_2')).select('combo_id', 'date', 'min', 'avg', 'max').repartition('date').cache()

    stocks_df = stocks_df.withColumn('price', f.col('price').cast(FloatType())).repartition('date')

    big_df = stocks_df.alias('a').join(df.alias('b'), f.col('a.date') == f.col('b.date'), 'inner').repartition(
        'stock_name', 'combo_id')
    # big_df = big_df.withColumn("cosine_sim", f.lit(1.0).cast("float"))
    big_df = big_df.select('a.date', 'stock_name', 'combo_id', 'price', 'avg', 'min', 'max').cache()

    df.unpersist()
    stocks_df.unpersist()

    print("Big DF size: ", big_df.count())
    bigdf_grouped = big_df.groupBy(["stock_name", 'combo_id']).agg(f.collect_list("price").alias('price'),
                                                                   f.collect_list("avg").alias('avg'),
                                                                   f.collect_list("min").alias('min'),
                                                                   f.collect_list("max").alias('max'))
    big_df.unpersist()

    def cosine(a, b):
        return float(np.dot(np.array(a), np.array(b)) / (np.linalg.norm(np.array(a)) * np.linalg.norm(np.array(b))))

    cosine_udf = f.udf(cosine, FloatType())
    df_final = bigdf_grouped.select('stock_name', 'combo_id', cosine_udf('price', 'max').alias('price_max_cosine'),
                                    cosine_udf('price', 'min').alias('price_min_cosine'),
                                    cosine_udf('price', 'avg').alias('price_avg_cosine'))

    bigdf_grouped.unpersist()

    print(df_final.show())


    print("No. of results (max): ", df_final.filter(df_final.price_max_cosine > 0.90).count())
    print("No. of results (min): ", df_final.filter(df_final.price_min_cosine > 0.90).count())
    print("No. of results (avg): ", df_final.filter(df_final.price_avg_cosine > 0.90).count())


if __name__ == "__main__":
    main()
