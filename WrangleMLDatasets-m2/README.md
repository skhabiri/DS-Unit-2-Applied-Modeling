**WrangleMLDatasets-232:** 

We use a large dataset from instacart file. The file is a tar.gz hosted on aws website. We use requests library to download the file and tarfile library to untar the file. It contantains multiple csv files like a relational database. Products.csv, orders.csv, order_products__train.csv, departments.csv, aisles.csv, order_products__prior.csv. The orders.csv has 3.4M rows with 7 columns. The ML problem that is defined to be tackled is what is the most frequently ordered product and how often is this product included in a customer's next order? Hence, it becomes a binary classification problem. Order_products__train.csv has 1.3M rows with 4 columns, “order_id”, “product_id”, “add_to_cart_order”, “reordered”. Products.csv has 49K rows and 4 columns: “product_id”, “product_name”, “aisle_id”, “department_id”. `pd.merge(order_products__train, products, how='inner', left_on='product_id', right_on='product_id')` creates a train set with order information, and product names. In the train dataset we have one row for each ordered product. In other words, there are multiple rows for the same order but different ordered products. 
`train['product_name'].value_counts()` shows banana is the most ordered item. A boolean column “banana” is created that says whether the product row is a banana or not. To understand how many of the orders contained banana we groupby based on order_id and count or sum “banana” feature, like 

`(train.groupby(by=['order_id']).sum())['bananas'].value_counts(normalize=True)` That shows 14% of the ordered included banana. To get details of any order like, hour_of_the_order, days_since_repeated_order, we merge orders and train dataframes. Then we can see what hour of the day the banana was ordered the most, `train2[train2['bananas']]['order_hour_of_day'].value_counts(normalize=True)` We use `train.groupby(['order_id']).count()['product_id'].mean()` to find average number of products per order. We can find out the average number of products ordered where banana was one of them, `train[train.order_id.isin(banana_order_ids)].groupby(['order_id']).count()['product_id'].mean()`

**Libraries:**
```
import requests
import tarfile
from glob import glob
import pandas as pd
from IPython.display import display
import numpy as np
import seaborn as sns
```

https://github.com/skhabiri/PredictiveModeling-AppliedModeling-u2s3/tree/master/WrangleMLDatasets-m2
