Sketchy is a very early version of a utility function I am developing to profile a Spark dataframe. The goal is to enable iterative "digging in" to understand a new and unfamiliar data set , although it is likely it could also be used for anomaly detection, software development testing, data quality auditing, and to analyze statistical distributions of datasets intended to be fed to machine learning models.

To use Sketchy, you would import the `profile` function from `sketchy` into PySpark, eg, `from sketchy import profile`. Then, you call profile on a dataframe like so:

```
from sketchy import profile

t = spark.table('my_namespace.my_table')

profile(t)
```

Sketchy is extremely immature -- I've spent about an hour hacking on it. I plan to spend more time working on it, and will keep this repo up to date as I develop it further. PRs to this repo are absolutely welcome.

There are a number of things I intend to improve about Sketchy in the near term, namely:

1. Support for profiling Arrays, Maps, Structs, and complex nested combinations of objects
2. Support for sampling a dataframe, preferably dynamically based on estimated input size
3. Support for enabling or disabling caching for performance; performance improvements overall
4. Support for varying degrees of detail when profiling a dataframe (eg percentiles, skewness of keys)
5. Unit tests for Spark data types

Longer term, I plan to experiment with comparing two data frames to each other to determine their similarity. Such a comparision, if reasonably simple to specify, could be used for regression testing of new code, data quality auditing over time, anamolly detection, and statistical distribution analysis of datasets that are about to be passed into a Machine Learning model or downstream ETL job.

If you have any ideas on how to improve Sketchy, feel free to open an issue on this repo to discuss it. Likewise, feel free to submit PRs or otherwise make suggestions about how to improve Sketchy.
