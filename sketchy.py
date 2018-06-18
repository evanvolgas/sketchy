import pyspark.sql.types as pst

from pyspark.sql import functions as F
from pyspark.sql import Row, SparkSession


spark = SparkSession.builder.appName('Test').getOrCreate()

SCHEMA = pst.StructType([
    pst.StructField('desc', pst.StringType(), True),
    pst.StructField('count', pst.IntegerType(), True),
    pst.StructField('min', pst.DoubleType(), True),
    pst.StructField('max', pst.DoubleType(), True),
    pst.StructField('sum', pst.DoubleType(), True),
    pst.StructField('stddev', pst.DoubleType(), True),
    pst.StructField('max_length', pst.IntegerType(), True),
    pst.StructField('cardinality', pst.IntegerType(), True),
])


def profile(df):
    x = spark.createDataFrame(sc.emptyRDD(), SCHEMA)
    df.cache()
    for field in df.schema.fields:
        if isinstance(field.dataType, (pst.StructField, pst.ArrayType, pst.MapType, pst.StructType, pst.BooleanType)):
            continue
        else:
            y = df.agg(
                F.lit(field.name).alias("desc"),
                F.count(field.name).alias('count'),
                F.min(field.name).alias('min'),
                F.max(field.name).alias('max'),
                F.sum(field.name).alias('sum'),
                F.stddev(field.name).alias('stddev'),
                F.max(F.length(field.name)).alias('max_length'),
                F.approx_count_distinct(field.name).alias('cardinality'))
            x = x.union(y)
    return x


df = spark.read.json("people.json")

profile(df).show(100, 100)
