# import findspark
# findspark.init()

import time
import os

import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext

sc = SparkContext()
ssc = StreamingContext(sc, 2)

#Checkpoint
ssc.checkpoint("checkpoint")
lines = ssc.socketTextStream("stream-host", 9000)

import numpy as np
from scipy import sparse

def get_vals(w: int = 5):
    coefs = np.random.randint(1, 1000, (w,w))
    p = np.random.randint(3000, 100000, w)
    return coefs, p + ~(p % 2).astype(bool)

def fit_poly(row, x):
    return np.poly1d(row)(x)

def cm_multihash(x: int, counts: int, coefs, p, const: int = 2719):
    table = np.zeros((p.shape[0], const))
    j = ((np.apply_along_axis(fit_poly, 1, coefs, x) % p) % const).astype(int)
    table[np.arange(0, p.shape[0]),j] = counts
    return sparse.csr_matrix(table)

def sparse_sim(sk1: sparse.csr.csr_matrix, sk2: sparse.csr.csr_matrix):
    return sk1.multiply(sk2).sum(axis = 1).min()

coefs, p = get_vals()

tau = 3000

def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder\
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]

def getUpdatesCounter(sparkContext):
    if ("counter" not in globals()):
        globals()["counter"] = sparkContext.accumulator(0)
    return globals()["counter"]

def tmp_fun(a,b):
    return a+b

def inv_fun(a,b):
    return a-b

def counting(time, rdd):
    counter = getUpdatesCounter(rdd.context)
    counter.add(rdd.count())


def process_data(time, rdd):
    # print("========= %s =========" % str(time))
    counter = getUpdatesCounter(rdd.context)
    # counter.add(rdd.count())
    collect = rdd.map(lambda line: line.split(',')) \
        .map(lambda line: (line[0], cm_multihash(int(line[1]), 1, coefs, p))) \
        .reduceByKey(tmp_fun) \
        .map(lambda line: (1, [line[0], line[1]]))
    fin = collect.join(collect).map(lambda row: [row[1][0][0], row[1][1][0], row[1][0][1], row[1][1][1]]) \
        .filter(lambda line: line[0] > line[1]) \
        .map(lambda line: [line[0], line[1], sparse_sim(line[-2], line[-1])]) \
        .filter(lambda line: line[2] >= tau).count()

    print(f'>>>> {str(time)[-8:]}  =====  {fin}  =====  {counter.value}')

lines.foreachRDD(counting)
lines.window(60).foreachRDD(process_data)

ssc.start()
ssc.awaitTerminationOrTimeout(300)
ssc.stop()
