**pdp-shaply-234:**

Partial dependence plots show the relationship between 1-2 individual features and the target — how predictions partially depend on the isolated features. For the pdp plot we use Lending club dataset, to predict the interest rate. Based on the target label it’s a regression problem. Data wrangling and feature engineering includes, selecting loans with ‘term’=36month. ‘issue_d’ is changed to datetime object and set as index. Some columns are renamed, “Int_rate” is changed from string to float type, new feature “monthly_debt” is engineered, and 6 columns are selected for modeling. In the selected column dataset all the rows with any null are dropped. The data is sliced based on the ‘issue_d’ index, with 10k val, 10k test, and the rest as train set. Each set is split into X, y where target is “Interest Rate”. The target distribution is checked to ensure it doesn’t have any skew. All the selected columns except “Loan Purpose” are numeric. We use Target encoder to make that feature numeric. In Target encoder, y can be binary or continuous. For each distinct element in categorical feature X you’re going to compute the average of the corresponding values in y. Then you’re going to replace each categorical value with the its y mean. With a pipeline of TragetEncoder() and LinearRegression() we achieve a R2 score of 0.22. For explaining the model we look at the coefficients of the LinearRegression. Scatter plot of individual features vs y is not particularly enlightening. We try OrdinalEncoder(). And XGBRegressor(). R2 score of validation set reaches 0.25. Unlike linear regressor the tree based regressor does not have coefficient to see the effect of each feature. Hence we use partial dependence plot to interpret the model. The values of a selected feature is replaced with a grid of sample points and the rest of the features carry the original values from val set. A plot of y vs feature shows the partial dependency of the target to the feature. pdp_isolate() is for single feature and pdp_interact() is for a list of features. The latter one can be visualized with a contour or heatmap (for two features). Pivoting the two features into index and column and having the predicted column as the value, we use plotly.graph_objs.Surface() for 3Dplot of two features. Next we use Titanic dataset to run pdp_plot() for a categorical feature (“sex”). For this we separate the encoder from the classifier in the pipeline and use the pdp_isolate() with the classifier alone. pdpbox does not bind the grid sample to discrete values of the encoded feature and we’ll use encoder mapping method to set the xtick with the categorical values instead of encoded numbers. Similar approach can be used with pdp_interact() for categorical features, to see dependency of target to more than one feature. 

If we want to see the effect of every feature in a single query (local effect) we use shapely for an additive explanation to see how each feature adds up. NYC apartment rent dataset is loaded. The target is “Price”. With LinearRegression() we can see the coefficients of the input features, in a linear regression. With RandomizedSearchCV() we perform a hyperparameter tuning on RandomForestRegressor(). We use shap.TreeExplainer(best_estimator_) to get the shap_values for a local query. The shap_values show in what way and how much the value of each feature contributed to the predicted value. Similarly we use Lending club data for a binary classification and use shap to interpret the model. There are two set of data, history and current with slightly different features. The history has a target as “loan_status” with two values of Fully Paid or Charged Off. history data is stratified and split into train, 20k val, and 20k test and each one of them into X and y. We use set operation -, & to get the exclusive and common columns of both datasets. Only features that are common between two datasets are selected. After wrangling the X_train, X_val, and X_test we have 66 columns and 80k for train and 20k rows for each of val and test. Next in a pipeline composed of OrdinalEncoder() and SimpleImputer() we X_train is fittransform-ed and X_val and X_test are transform-ed. Then we fit XGBClassifier() on the transformed train data and evaluate the transformed eval data. `XGBClassifier(n_estimators=1000, n_jobs=-1).fit(X_train_processed, y_train, eval_set=eval_set, eval_metric='auc', early_stopping_rounds=10)`. Applying predict_proba() and roc_auc_score() method we get ROC-AUC of 0.7 for transformed test data. In a similar way we use shap.TreeExplainer(model) and explainer.shap_values(row_processed) where row_processed is the transformed row with the pipeline. And then we use shap.force_plot() to plot the local additive dependency, shap to interpret the prediction for the row query.

**Libraries:**
```
import sys
import warnings
import pandas as pd
import seaborn as sns
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from pdpbox.pdp import pdp_interact, pdp_interact_plot
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from pdpbox import pdp
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import shap
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
```
https://github.com/skhabiri/PredictiveModeling-AppliedModeling-u2s3/tree/master/ModelInterpretation-m4

