import pyspark.sql.types as pst

from pyspark.sql import functions as F
from pyspark.sql import Row, SparkSession

spark = SparkSession.builder.appName('Sketchy').getOrCreate()

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


def profile(df, cache=True):
    x = spark.createDataFrame(sc.emptyRDD(), SCHEMA)
    if cache:
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


"""
Using a static CSV Download of https://data.lacity.org/A-Safe-City/Crime-Data-from-2010-to-Present/y8tr-7khq, you can invoke sketchy as follows

from sketchy import profile

df = spark.read.csv('Crime_Data_from_2010_to_Present.csv', header=True)

In [7]: profile(df).show(100, 100)
+----------------------+-------+----------------------------------------+-------------------------------+-------------------+--------------------+----------+-----------+
|                  desc|  count|                                     min|                            max|                sum|              stddev|max_length|cardinality|
+----------------------+-------+----------------------------------------+-------------------------------+-------------------+--------------------+----------+-----------+
|             DR Number|1759630|                               001208575|                      910220366|2.46170762846376E14|2.4764794511086047E7|         9|    1733518|
|         Date Reported|1759630|                              01/01/2010|                     12/31/2017|               null|                null|        10|       3121|
|         Date Occurred|1759630|                              01/01/2010|                     12/31/2017|               null|                null|        10|       3121|
|         Time Occurred|1759630|                                    0001|                           2359|      2.398564757E9|   646.6303202961955|         4|       1450|
|               Area ID|1759630|                                      01|                             21|        1.9596122E7|   5.994906682126726|         2|         21|
|             Area Name|1759630|                             77th Street|                       Wilshire|               null|                null|        11|         21|
|    Reporting District|1759630|                                    0100|                           2198|      2.041220665E9|   599.4875316669362|         4|       1349|
|            Crime Code|1759630|                                     110|                            956|       8.92014447E8|  210.54419314084458|         3|        142|
|Crime Code Description|1759244|                        ABORTION/ILLEGAL|     WEAPONS POSSESSION/BOMBING|               null|                null|        63|        154|
|              MO Codes|1569068|                                    0100|                 9999 2004 0910|       2.90772261E8|   535.5456869915018|        49|     392197|
|            Victim Age|1618872|                                      10|                             99|         5.826381E7|  16.808113208934202|         2|         94|
|            Victim Sex|1597370|                                       -|                              X|               null|                null|         1|          5|
|        Victim Descent|1597330|                                       -|                              Z|               null|                null|         1|         19|
|          Premise Code|1759544|                                     101|                            971|       5.48818586E8|  210.70621133103677|         3|        318|
|   Premise Description|1756374|7TH AND METRO CENTER (NOT LINE SPECIFIC)|    YARD (RESIDENTIAL/BUSINESS)|               null|                null|        63|        316|
|      Weapon Used Code| 584439|                                     101|                            516|       2.16801071E8|  113.76864686575261|         3|         82|
|    Weapon Description| 584438|        AIR PISTOL/REVOLVER/RIFLE/BB GUN|                  VERBAL THREAT|               null|                null|        46|         74|
|           Status Code|1759628|                                      13|                             TH|               32.0|   4.242640687119285|         2|          9|
|    Status Description|1759630|                            Adult Arrest|                            UNK|               null|                null|        12|          6|
|          Crime Code 1|1759624|                                     110|                            999|       8.91775023E8|   210.4486394938767|         3|        150|
|          Crime Code 2| 112315|                                     210|                            999|       1.06984939E8|   124.9594359399203|         3|        146|
|          Crime Code 3|   2589|                                     310|                            999|          2513499.0|   87.86962966937037|         3|         59|
|          Crime Code 4|     86|                                     421|                            999|            83545.0|   88.43098431934419|         3|         11|
|               Address|1759630|                                      00|ZOO                          DR|              850.0|  47.521064938648266|        40|      71472|
|          Cross Street| 292495|         10                           FY|ZUNIGA                       LN|           663143.0|   234028.9674728552|        34|      12383|
|             Location |1759621|                                  (0, 0)|            (34.7907, -118.317)|               null|                null|        20|      61670|
+----------------------+-------+----------------------------------------+-------------------------------+-------------------+--------------------+----------+-----------+


"""
