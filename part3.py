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

sc = SparkContext(conf=conf).getOrCreate()
ss = SparkSession.builder.config(conf=conf).appName("Milestone 1 - Group 29").getOrCreate()

def main():
    print("********* MS1 - part 3 ************")
    weather_df = ss.read.option('header', True).csv("weather-ta2.csv").limit(50000)
    stocks_df = ss.read.option('header', True).csv("stocks-ta2.csv").limit(50000)

    weather = weather_df.rdd
    stocks = stocks_df.rdd

    weather = weather.map(lambda x: [x[1], x[2], x[3]])
    stocks = stocks.map(lambda x: [x[1], x[2], x[3]])

    df1 = weather.keyBy(lambda row: row[1])

    def comboid_create(row):
        el1 = row[1][0]
        el2 = row[1][1]
        comboid = [el1[0], el2[0]]
        comboid.sort()
        return ['_'.join(comboid), el1[1], el1[2], el2[2]]

    df2 = df1.join(df1).filter(lambda row: row[1][0][0] > row[1][1][0]).map(comboid_create)

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

    def aggregate(row):
        a = row[-2]
        b = row[-1]
        return [row[0], row[1], min(a, b), avg(a, b), max(a, b)]

    df3 = df2.map(aggregate)
    stocks = stocks.map(lambda row: [row[0], row[1], float(row[2])]).keyBy(lambda row: row[1])
    bigdata = stocks.join(df3.keyBy(lambda row: row[1]))

    def parse_after_join(row):
        el1 = row[1][0]
        el2 = row[1][1]
        return [el1[0], el2[0], [el1[-1]], [el2[-3]], [el2[-2]], [el2[-1]]]

    bigdata = bigdata.map(parse_after_join)
    agg = bigdata.keyBy(lambda row: (row[0], row[1])).reduceByKey(
        lambda a, b: [a[-4] + b[-4], a[-3] + b[-3], a[-2] + b[-2], a[-1] + b[-1]])

    def cosine(a, b):
        return float(np.dot(np.array(a), np.array(b)) / (np.linalg.norm(np.array(a)) * np.linalg.norm(np.array(b))))

    def parser(key):
        return key[:-12], key[-11:]

    def get_cosine(tuple_row):
        price_min = cosine(tuple_row[1][0], tuple_row[1][1])
        price_avg = cosine(tuple_row[1][0], tuple_row[1][2])
        price_max = cosine(tuple_row[1][0], tuple_row[1][3])
        stock, combo_id = tuple_row[0]
        station1, station2 = parser(combo_id)
        return [stock, station1, station2, price_min, price_avg, price_max]

    results = agg.map(get_cosine)

    tau = [0.90, 0.90, 0.90]

    min_results = results.filter(lambda row: row[-3] > tau[0]).collect()
    avg_results = results.filter(lambda row: row[-2] > tau[1]).collect()
    max_results = results.filter(lambda row: row[-1] > tau[2]).collect()

    print("Results =>")
    print(f"Min Results(count={len(min_results)}): ", min_results)
    print(f"Avg Results(count={len(avg_results)}): ", avg_results)
    print(f"Max Results(count={len(max_results)}): ", max_results)


if __name__ == "__main__":
    main()