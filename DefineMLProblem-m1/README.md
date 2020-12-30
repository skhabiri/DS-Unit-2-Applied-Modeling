**DefineMLProblem - 231:**

We used Burrito dataset to apply machine learning model. First step was to choose a target label. We created a two class target label “Great” based on a column labeled “overall” which was the customer rating from 1 to 5. Next we did exploratory data analysis EDA and data wrangling to clean up the data. In this phase we would consider the following steps. For meaningful variation: Are there any features with quasi constant values that could be dropped. Any duplicate rows or columns, highly correlated features, high cardinality features. For categorical features: consider dropping high cardinality categorical features, grouping rare labels, any categorical feature that could be ranked like ordinal encoding. For Outlier: if the model needs to be less sensitive to outliers we could choose a decision tree based model. We could filter out the outlier percentile of certain features, or model the logarithm of the features log1p() to eliminate the skewness, Linear models, neural networks, and other distance-based models will almost always benefit from scaling, and normalization. For feature selection: we need to reduce the number of features to reduce the chance of overfitting, facilitate model interpretation for stakeholders, reduce computational resources, facilitate implementation, prevent leakage from target label to input of the model. Next step we use the “Date” column to split the data into train, validation and test sets. We use value_counts() to establish the baseline for our model. As for distribution: for classification problems, how many classes in the target label, and if the classes are balanced or imbalanced, for a linear regression problem we would need the target label to be normally distributed, if the target is right-skewed, we may want to log transform the target. For reproducibility: set a random seed, wrap the code in reproducible function/classes for modularity of steps, including feature loading, data wrangling, feature processing, etc. Combine your modularized functions / classes in a single, centralized pipeline to execute. In choosing metrics: for classification problems we can use accuracy for balanced target label and also use precision, recall, roc-auc, f1-score for imbalance target labels. For regression problems, we use mae, mse, r2 score.

*Libraries:*
```
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import category_encoders as ce
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
from sklearn.metrics import plot_confusion_matrix, classification_report
```

https://github.com/skhabiri/PredictiveModeling-AppliedModeling-u2s3/tree/master/DefineMLProblem-m1
