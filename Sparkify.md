
# Sparkify Project Workspace
This workspace contains a tiny subset (128MB) of the full dataset available (12GB). Feel free to use this workspace to build your project, or to explore a smaller subset with Spark before deploying your cluster on the cloud. Instructions for setting up your Spark cluster is included in the last lesson of the Extracurricular Spark Course content.

You can follow the steps below to guide your data analysis and model building portion of this project.


```python
# import libraries
from pyspark.sql import SparkSession, SQLContext , Window
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf, count,sum as Fsum
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import StandardScaler,RegexTokenizer, StringIndexer, CountVectorizer, IDF, VectorAssembler, Normalizer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator , MulticlassClassificationEvaluator
from time import time

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import pandas as pd


```


```python
# create a Spark session
spark = SparkSession.builder.master("local").appName("Capstone_Project").getOrCreate()
```

# Load and Clean Dataset
In this workspace, the mini-dataset file is `mini_sparkify_event_data.json`. Load and clean the dataset, checking for invalid or missing data - for example, records without userids or sessionids. 


```python
# load data into spark DataFrame

mydata = spark.read.json("./mini_sparkify_event_data.json")

mydata.printSchema()
```

    root
     |-- artist: string (nullable = true)
     |-- auth: string (nullable = true)
     |-- firstName: string (nullable = true)
     |-- gender: string (nullable = true)
     |-- itemInSession: long (nullable = true)
     |-- lastName: string (nullable = true)
     |-- length: double (nullable = true)
     |-- level: string (nullable = true)
     |-- location: string (nullable = true)
     |-- method: string (nullable = true)
     |-- page: string (nullable = true)
     |-- registration: long (nullable = true)
     |-- sessionId: long (nullable = true)
     |-- song: string (nullable = true)
     |-- status: long (nullable = true)
     |-- ts: long (nullable = true)
     |-- userAgent: string (nullable = true)
     |-- userId: string (nullable = true)
    


## Clean Data


```python
#Clean Dataset

# temp view of the data frame

mydata.createOrReplaceTempView('data_tbl')


```


```python
# check if there are nulls in sessionId column

spark.sql("""
            SELECT COUNT(userId) as UserId
            FROM data_tbl
            WHERE sessionId IS NULL
        """).show()
```

    +------+
    |UserId|
    +------+
    |     0|
    +------+
    



```python
# check if there are empty sessionIds

spark.sql("""
            SELECT COUNT(userId) as UserId
            FROM data_tbl
            WHERE sessionId == ''
        """).show()
```

    +------+
    |UserId|
    +------+
    |     0|
    +------+
    



```python
# check if there are nulls in userId column

spark.sql("""
            SELECT COUNT(userId) as UserId
            FROM data_tbl
            WHERE userId IS NULL
        """).show()
```

    +------+
    |UserId|
    +------+
    |     0|
    +------+
    



```python
# check if there are empty UserIDs

spark.sql("""
            SELECT COUNT(userId) as UserId
            FROM data_tbl
            WHERE userId == ''
        """).show()
```

    +------+
    |UserId|
    +------+
    |  8346|
    +------+
    



```python
# remove the invalid user IDs from the dataset

mydata = spark.sql("""
                    SELECT *
                    FROM data_tbl
                    WHERE userId != ''
                """)
```


```python
# temporary view of the data frame

mydata.createOrReplaceTempView('data_tbl')
```

# Exploratory Data Analysis
When you're working with the full dataset, perform EDA by loading a small subset of the data and doing basic manipulations within Spark. In this workspace, you are already provided a small subset of data you can explore.

### Define Churn

Once you've done some preliminary analysis, create a column `Churn` to use as the label for your model. I suggest using the `Cancellation Confirmation` events to define your churn, which happen for both paid and free users. As a bonus task, you can also look into the `Downgrade` events.



```python
page = mydata.select("page").dropDuplicates().show()

```

    +--------------------+
    |                page|
    +--------------------+
    |              Cancel|
    |    Submit Downgrade|
    |         Thumbs Down|
    |                Home|
    |           Downgrade|
    |         Roll Advert|
    |              Logout|
    |       Save Settings|
    |Cancellation Conf...|
    |               About|
    |            Settings|
    |     Add to Playlist|
    |          Add Friend|
    |            NextSong|
    |           Thumbs Up|
    |                Help|
    |             Upgrade|
    |               Error|
    |      Submit Upgrade|
    +--------------------+
    



```python
# create churn user list

mydata = spark.sql("""
                    SELECT *,
                           CASE
                                WHEN page == 'Cancellation Confirmation' THEN 1
                                ELSE 0 END as Churned
                    FROM data_tbl
                """)

mydata.createOrReplaceTempView('data_tbl')

Churned = spark.sql("""
                            SELECT DISTINCT userID
                            FROM data_tbl
                            WHERE Churned = 1
                        """).toPandas().values

Churned = [user[0] for user in Churned]

```


```python
#show churned and non-churned user in dataset

spark.sql("""
          SELECT
              Churned,
              count(distinct userId)
            FROM
                data_tbl
            GROUP BY
                Churned
            """)

```




    DataFrame[Churned: int, count(DISTINCT userId): bigint]




```python
#create churn table

churn = spark.sql("""
          SELECT
              distinct userId,
              Churned
            FROM
                data_tbl

            """)
churn.createOrReplaceTempView('churn')

```


```python
# show  churn in gender

spark.sql("""
          SELECT distinct
              gender,
              Churned,
              count(distinct userId) as DistinctUsers
            FROM
                data_tbl
            GROUP BY
                gender,Churned
            order by Churned desc
            """)
```




    DataFrame[gender: string, Churned: int, DistinctUsers: bigint]



### Explore Data
Once you've defined churn, perform some exploratory data analysis to observe the behavior for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.


```python
explore_data = mydata.toPandas()
```


```python
explore_data.drop('Churned',axis='columns', inplace=True)
```


```python
explore_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>auth</th>
      <th>firstName</th>
      <th>gender</th>
      <th>itemInSession</th>
      <th>lastName</th>
      <th>length</th>
      <th>level</th>
      <th>location</th>
      <th>method</th>
      <th>page</th>
      <th>registration</th>
      <th>sessionId</th>
      <th>song</th>
      <th>status</th>
      <th>ts</th>
      <th>userAgent</th>
      <th>userId</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sleeping With Sirens</td>
      <td>Logged In</td>
      <td>Darianna</td>
      <td>F</td>
      <td>0</td>
      <td>Carpenter</td>
      <td>202.97098</td>
      <td>free</td>
      <td>Bridgeport-Stamford-Norwalk, CT</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1538016340000</td>
      <td>31</td>
      <td>Captain Tyin Knots VS Mr Walkway (No Way)</td>
      <td>200</td>
      <td>1539003534000</td>
      <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like...</td>
      <td>100010</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Francesca Battistelli</td>
      <td>Logged In</td>
      <td>Darianna</td>
      <td>F</td>
      <td>1</td>
      <td>Carpenter</td>
      <td>196.54485</td>
      <td>free</td>
      <td>Bridgeport-Stamford-Norwalk, CT</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1538016340000</td>
      <td>31</td>
      <td>Beautiful_ Beautiful (Album)</td>
      <td>200</td>
      <td>1539003736000</td>
      <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like...</td>
      <td>100010</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brutha</td>
      <td>Logged In</td>
      <td>Darianna</td>
      <td>F</td>
      <td>2</td>
      <td>Carpenter</td>
      <td>263.13098</td>
      <td>free</td>
      <td>Bridgeport-Stamford-Norwalk, CT</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1538016340000</td>
      <td>31</td>
      <td>She's Gone</td>
      <td>200</td>
      <td>1539003932000</td>
      <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like...</td>
      <td>100010</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Darianna</td>
      <td>F</td>
      <td>3</td>
      <td>Carpenter</td>
      <td>NaN</td>
      <td>free</td>
      <td>Bridgeport-Stamford-Norwalk, CT</td>
      <td>PUT</td>
      <td>Thumbs Up</td>
      <td>1538016340000</td>
      <td>31</td>
      <td>None</td>
      <td>307</td>
      <td>1539003933000</td>
      <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like...</td>
      <td>100010</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Josh Ritter</td>
      <td>Logged In</td>
      <td>Darianna</td>
      <td>F</td>
      <td>4</td>
      <td>Carpenter</td>
      <td>316.23791</td>
      <td>free</td>
      <td>Bridgeport-Stamford-Norwalk, CT</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1538016340000</td>
      <td>31</td>
      <td>Folk Bloodbath</td>
      <td>200</td>
      <td>1539004195000</td>
      <td>"Mozilla/5.0 (iPhone; CPU iPhone OS 7_1_2 like...</td>
      <td>100010</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
explore_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>itemInSession</th>
      <th>length</th>
      <th>registration</th>
      <th>sessionId</th>
      <th>status</th>
      <th>ts</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>278154.000000</td>
      <td>228108.000000</td>
      <td>2.781540e+05</td>
      <td>278154.000000</td>
      <td>278154.000000</td>
      <td>2.781540e+05</td>
      <td>278154.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>114.899182</td>
      <td>249.117182</td>
      <td>1.535359e+12</td>
      <td>1042.561624</td>
      <td>209.103216</td>
      <td>1.540959e+12</td>
      <td>0.161292</td>
    </tr>
    <tr>
      <th>std</th>
      <td>129.851729</td>
      <td>99.235179</td>
      <td>3.291322e+09</td>
      <td>726.501036</td>
      <td>30.151389</td>
      <td>1.506829e+09</td>
      <td>0.367801</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.783220</td>
      <td>1.521381e+12</td>
      <td>1.000000</td>
      <td>200.000000</td>
      <td>1.538352e+12</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>199.888530</td>
      <td>1.533522e+12</td>
      <td>338.000000</td>
      <td>200.000000</td>
      <td>1.539699e+12</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>70.000000</td>
      <td>234.500770</td>
      <td>1.536664e+12</td>
      <td>1017.000000</td>
      <td>200.000000</td>
      <td>1.540934e+12</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>157.000000</td>
      <td>277.158730</td>
      <td>1.537672e+12</td>
      <td>1675.000000</td>
      <td>200.000000</td>
      <td>1.542268e+12</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1321.000000</td>
      <td>3024.665670</td>
      <td>1.543247e+12</td>
      <td>2474.000000</td>
      <td>404.000000</td>
      <td>1.543799e+12</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
explore_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 278154 entries, 0 to 278153
    Data columns (total 19 columns):
    artist           228108 non-null object
    auth             278154 non-null object
    firstName        278154 non-null object
    gender           278154 non-null object
    itemInSession    278154 non-null int64
    lastName         278154 non-null object
    length           228108 non-null float64
    level            278154 non-null object
    location         278154 non-null object
    method           278154 non-null object
    page             278154 non-null object
    registration     278154 non-null int64
    sessionId        278154 non-null int64
    song             228108 non-null object
    status           278154 non-null int64
    ts               278154 non-null int64
    userAgent        278154 non-null object
    userId           278154 non-null object
    churn            278154 non-null int64
    dtypes: float64(1), int64(6), object(12)
    memory usage: 40.3+ MB



```python
# from checking if there are empty UserIDs above
# we will drop empty values
explore_data = explore_data.filter(explore_data.userId != '')
```


```python
cancelation_flag = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
```


```python
explore_data = mydata.withColumn("churn", cancelation_flag("page"))
window_value = Window.partitionBy("userId").rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)
```


```python
explore_data = explore_data.withColumn("churn", Fsum("churn").over(window_value))
```


```python
explore_data = explore_data.toPandas()
```


```python
explore_data.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>auth</th>
      <th>firstName</th>
      <th>gender</th>
      <th>itemInSession</th>
      <th>lastName</th>
      <th>length</th>
      <th>level</th>
      <th>location</th>
      <th>method</th>
      <th>page</th>
      <th>registration</th>
      <th>sessionId</th>
      <th>song</th>
      <th>status</th>
      <th>ts</th>
      <th>userAgent</th>
      <th>userId</th>
      <th>Churned</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>187258</th>
      <td>The xx</td>
      <td>Logged In</td>
      <td>Elias</td>
      <td>M</td>
      <td>30</td>
      <td>Love</td>
      <td>313.39057</td>
      <td>paid</td>
      <td>Salinas, CA</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1532696273000</td>
      <td>348</td>
      <td>Infinity</td>
      <td>200</td>
      <td>1542409487000</td>
      <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebK...</td>
      <td>200025</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>52080</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Brayden</td>
      <td>M</td>
      <td>30</td>
      <td>Thomas</td>
      <td>NaN</td>
      <td>paid</td>
      <td>Los Angeles-Long Beach-Anaheim, CA</td>
      <td>GET</td>
      <td>Home</td>
      <td>1534133898000</td>
      <td>734</td>
      <td>None</td>
      <td>200</td>
      <td>1539320124000</td>
      <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4...</td>
      <td>85</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>207939</th>
      <td>Carl Dobkins_ Jr.</td>
      <td>Logged In</td>
      <td>Alexander</td>
      <td>M</td>
      <td>92</td>
      <td>Garcia</td>
      <td>120.21506</td>
      <td>paid</td>
      <td>Indianapolis-Carmel-Anderson, IN</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1536817381000</td>
      <td>508</td>
      <td>My Heart Is An Open Book</td>
      <td>200</td>
      <td>1539325590000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>105</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>270649</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Jayden</td>
      <td>M</td>
      <td>35</td>
      <td>Santos</td>
      <td>NaN</td>
      <td>free</td>
      <td>Dallas-Fort Worth-Arlington, TX</td>
      <td>GET</td>
      <td>Help</td>
      <td>1533812833000</td>
      <td>67</td>
      <td>None</td>
      <td>200</td>
      <td>1539075198000</td>
      <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10)...</td>
      <td>100018</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>202384</th>
      <td>None</td>
      <td>Logged In</td>
      <td>Micah</td>
      <td>M</td>
      <td>67</td>
      <td>Long</td>
      <td>NaN</td>
      <td>paid</td>
      <td>Boston-Cambridge-Newton, MA-NH</td>
      <td>GET</td>
      <td>Settings</td>
      <td>1538331630000</td>
      <td>2334</td>
      <td>None</td>
      <td>200</td>
      <td>1543319750000</td>
      <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebK...</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>187651</th>
      <td>Lifehouse</td>
      <td>Logged In</td>
      <td>Andrew</td>
      <td>M</td>
      <td>161</td>
      <td>Poole</td>
      <td>259.89179</td>
      <td>paid</td>
      <td>Greensboro-High Point, NC</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1541223737000</td>
      <td>1719</td>
      <td>The End Has Only Begun</td>
      <td>200</td>
      <td>1541744505000</td>
      <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebK...</td>
      <td>153</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>250710</th>
      <td>The Verve</td>
      <td>Logged In</td>
      <td>Saul</td>
      <td>M</td>
      <td>4</td>
      <td>Johnson</td>
      <td>360.25424</td>
      <td>paid</td>
      <td>Houston-The Woodlands-Sugar Land, TX</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1531804365000</td>
      <td>1763</td>
      <td>Bitter Sweet Symphony</td>
      <td>200</td>
      <td>1542011203000</td>
      <td>Mozilla/5.0 (Windows NT 6.1; WOW64; rv:31.0) G...</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>129529</th>
      <td>P!nk</td>
      <td>Logged In</td>
      <td>Michael</td>
      <td>M</td>
      <td>270</td>
      <td>Miller</td>
      <td>227.02975</td>
      <td>paid</td>
      <td>Phoenix-Mesa-Scottsdale, AZ</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1537014411000</td>
      <td>1400</td>
      <td>Glitter In The Air</td>
      <td>200</td>
      <td>1541019841000</td>
      <td>Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; r...</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17678</th>
      <td>Rytmus</td>
      <td>Logged In</td>
      <td>Sadie</td>
      <td>F</td>
      <td>217</td>
      <td>Jones</td>
      <td>254.24934</td>
      <td>paid</td>
      <td>Denver-Aurora-Lakewood, CO</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1537054553000</td>
      <td>2065</td>
      <td>Ani jeden skurvy me nezastavi (Explicit)</td>
      <td>200</td>
      <td>1542832017000</td>
      <td>"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_4...</td>
      <td>132</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>219436</th>
      <td>Violadores del Verso</td>
      <td>Logged In</td>
      <td>Lauren</td>
      <td>F</td>
      <td>177</td>
      <td>Boone</td>
      <td>325.72036</td>
      <td>paid</td>
      <td>St. Louis, MO-IL</td>
      <td>PUT</td>
      <td>NextSong</td>
      <td>1534859694000</td>
      <td>76</td>
      <td>Nada mas</td>
      <td>200</td>
      <td>1539279730000</td>
      <td>"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebK...</td>
      <td>300009</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

# check churned and non churned users

explore_data.drop_duplicates(subset='userId').groupby(['churn'])['userId'].count()
```




    churn
    0    173
    1     52
    Name: userId, dtype: int64




```python
# Plot function to visualize some Data

def plot(subset, group, labels, x_title="Num. of users", y_title="Sub. Status"):
    
    ax = explore_data.drop_duplicates(subset=subset).groupby(group)['userId'].count().plot(kind='barh',title='Num. of users per category')
    
    ax.set_xlabel(x_title)
    ax.set_yticklabels(labels)
    ax.set_ylabel(y_title)
    
```


```python
plot(subset = ['userId'], group = ['churn'], labels = ['Active Users', 'Cancelled Users'])
```


![png](output_33_0.png)



```python
# check churned and non churned users by Gender

explore_data.drop_duplicates(subset='userId').groupby(['gender', 'churn'])['gender'].count()
```




    gender  churn
    F       0        84
            1        20
    M       0        89
            1        32
    Name: gender, dtype: int64




```python
plot(subset=['userId', 'gender'], group = ['gender', 'churn'], labels = ['Female-Active', 'Female_Cancelled', 'Male_Active', 'Female_Cancelled'])
```


![png](output_35_0.png)



```python
# check churned and non churned users by Payment

explore_data.drop_duplicates(subset='userId').groupby(['level', 'churn'])['level'].count()
```




    level  churn
    free   0        133
           1         44
    paid   0         40
           1          8
    Name: level, dtype: int64




```python
plot(subset=['userId', 'level'], group=['level', 'churn'], labels=['Free_Active', 'Free_Cancelled', 'Paid_Active', 'Paid_Cancelled'])
```


![png](output_37_0.png)


# Feature Engineering
Once you've familiarized yourself with the data, build out the features you find promising to train your model on. To work with the full dataset, you can follow the following steps.
- Write a script to extract the necessary features from the smaller subset of data
- Ensure that your script is scalable, using the best practices discussed in Lesson 3
- Try your script on the full data set, debugging your script if necessary

If you are working in the classroom workspace, you can just extract features based on the small subset of data contained here. Be sure to transfer over this work to the larger dataset when you work on your Spark cluster.


```python
#gender feature " replace str by int values"


gender = mydata.dropDuplicates(['userId']).sort('userId').select(['userId','gender'])
gender = gender.replace(['F','M'], ['1', '0'], 'gender')
gender = gender.withColumn('gender', gender.gender.cast("int"))

gender.createOrReplaceTempView('gender')
```


```python
#number of songs played per user 

songs = mydata.where(mydata.song!='null').groupby('userId')
songs= songs.agg(count(mydata.song).alias('Played_Songs')).orderBy('userId')
songs = songs.select(['userId','Played_Songs'])

songs.createOrReplaceTempView('songs')
```


```python
# number of listened singers per user 

listened_singers_per_user = mydata.dropDuplicates(['userId','artist']).groupby('userId')
listened_singers_per_user = listened_singers_per_user.agg(count(mydata.artist).alias('Listened_Singers')).orderBy('userId')
listened_singers_per_user = listened_singers_per_user.select(['userId','Listened_Singers'])

listened_singers_per_user.createOrReplaceTempView('listened_singers_per_user')
```


```python
#thumbs_Down

thumbs_Down = mydata.where(mydata.page=='Thumbs Down').groupby(['userId'])
thumbs_Down = thumbs_Down.agg(count(col('page')).alias('thumbs_down')).orderBy('userId')
thumbs_Down = thumbs_Down.select(['userId','thumbs_down'])

thumbs_Down.createOrReplaceTempView('thumbs_Down')
```


```python
#thumbs_Up

thumbs_Up = mydata.where(mydata.page=='Thumbs Up').groupby(['userId'])
thumbs_Up = thumbs_Up.agg(count(col('page')).alias('thumbs_Up')).orderBy('userId')
thumbs_Up = thumbs_Up.select(['userId','thumbs_Up'])

thumbs_Up.createOrReplaceTempView('thumbs_Up')
```

# Modeling
Split the full dataset into train, test, and validation sets. Test out several of the machine learning methods you learned. Evaluate the accuracy of the various models, tuning parameters as necessary. Determine your winning model based on test accuracy and report results on the validation set. Since the churned users are a fairly small subset, I suggest using F1 score as the metric to optimize.


```python
# join features

Data = churn.dropDuplicates(['userId']).sort('userId').select(['userId','Churned'])
for selected_features in [ gender, songs, listened_singers_per_user, thumbs_Up, thumbs_Down]:
    Data = Data.join(selected_features,'userId')

```


```python
# convert data type into float
for selected_features in Data.columns[1:]:
    Data = Data.withColumn(selected_features,Data[selected_features].cast('float'))
```


```python
Data.dtypes
```




    [('userId', 'string'),
     ('Churned', 'float'),
     ('gender', 'float'),
     ('Played_Songs', 'float'),
     ('Listened_Singers', 'float'),
     ('thumbs_Up', 'float'),
     ('thumbs_down', 'float')]




```python
# split our data into train and test sets

train_set, test_set = Data.randomSplit([0.8, 0.2])
```


```python
assembler = VectorAssembler(inputCols=Data.columns[2:],outputCol='featuresassemble')
scaler = StandardScaler(inputCol="featuresassemble", outputCol="features")
indexer = StringIndexer(inputCol="Churned", outputCol="label")
stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
RandomForestClassifier = RandomForestClassifier(numTrees=3, maxDepth=2, labelCol="indexed")
LogisticRegression =  LogisticRegression(maxIter=100, regParam=0.0, elasticNetParam=0)
```


```python
LogisticRegression_pipeline = Pipeline(stages=[assembler, scaler, indexer, LogisticRegression])

paramGrid_LogisticRegression = ParamGridBuilder().addGrid(LogisticRegression.regParam,[0.0, 0.1, 0.01]).build()

CrossValidator_LogisticRegression = CrossValidator(estimator=LogisticRegression_pipeline,estimatorParamMaps=paramGrid_LogisticRegression,
                                    evaluator=MulticlassClassificationEvaluator(metricName = 'f1'),numFolds=3)

start_time = time()
CrossValidator_LogisticRegression_Model = CrossValidator_LogisticRegression.fit(train_set)
end_time = time()

print('The training process take {} seconds'.format(end_time - start_time))

CrossValidator_LogisticRegression_Model.avgMetrics
```

    The training process take 923.1168491840363 seconds





    [0.8297728719764362, 0.8256805908727608, 0.8256805908727608]




```python
RandomForest_pipeline = Pipeline(stages=[assembler, scaler, indexer, stringIndexer, RandomForestClassifier])

paramGrid_RandomForest = ParamGridBuilder().addGrid(RandomForestClassifier.numTrees,[10, 30]).build()

CrossValidator_RandomForest = CrossValidator(estimator=RandomForest_pipeline,estimatorParamMaps=paramGrid_RandomForest,
                              evaluator=MulticlassClassificationEvaluator(metricName = 'f1'),numFolds=3)

start_time = time()
CrossValidator_RandomForest_Model = CrossValidator_RandomForest.fit(train_set)
end_time = time()

print('The training process take {} seconds'.format(end_time - start_time))

CrossValidator_RandomForest_Model.avgMetrics
```

    The training process take 617.9057967662811 seconds





    [0.8256805908727608, 0.8256805908727608]



# Performance of models



```python
def performance(model, test_data, metric = 'f1'):
    
    """
    
    this function to Evaluate model performance 
    
        Input: 
            model - trained model
            metric - used metric to evaluate performance
            data - test set that performance measurement should be performed
            
        Output:
            evaluated_score
    """
    
    evaluator = BinaryClassificationEvaluator(metricName = 'areaUnderROC')
    
    predictions = model.transform(test_data)
    
    # evaluated_score
    evaluated_score = evaluator.evaluate(predictions)
    
    return evaluated_score
```


```python
model_RandomForest_fitted = RandomForest_pipeline.fit(train_set)
model_LogisticRegression_fitted = LogisticRegression_pipeline.fit(train_set)
```


```python
performance(model_RandomForest_fitted, test_set)
```




    0.49107142857142855




```python
performance(model_LogisticRegression_fitted, test_set)
```




    0.7202380952380952



# Final Steps
Clean up your code, adding comments and renaming variables to make the code easier to read and maintain. Refer to the Spark Project Overview page and Data Scientist Capstone Project Rubric to make sure you are including all components of the capstone project and meet all expectations. Remember, this includes thorough documentation in a README file in a Github repository, as well as a web app or blog post.


```python

```
