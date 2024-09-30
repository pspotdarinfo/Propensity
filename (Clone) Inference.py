# Databricks notebook source


# COMMAND ----------

import pandas as pd
import joblib
import numpy as np

# COMMAND ----------

data=pd.read_csv('/Workspace/Users/pratik.potdar@infocepts.com/mlops/mlops_input_data.csv')

# COMMAND ----------

data.head()

# COMMAND ----------

data.drop(['Unnamed: 0'],axis=1,inplace=True)

# COMMAND ----------

data.isnull().sum()

# COMMAND ----------

data['total_value_remittance_deposits'].fillna(0, inplace=True)
data['total_num_remittance_deposits'].fillna(0, inplace=True)

# COMMAND ----------

data['day_since_last_withdrawal'] = data['day_since_last_withdrawal'].fillna(data['had_debcard'].map({0: 72, 1: 21}))

# COMMAND ----------

cat_cols=data.select_dtypes(include=['object'])


# COMMAND ----------

cat_cols.drop(['business_id'],axis=1,inplace=True)

# COMMAND ----------

cat_cols['operating_address_city'].fillna('unknown', inplace=True)

# COMMAND ----------

label_cols=['gender','nationality']

# COMMAND ----------

cat_cols['gender'].nunique()

# COMMAND ----------

label_encoders = {col: joblib.load(f'/Workspace/Users/pratik.potdar@infocepts.com/mlops/{col}_label_encoder.pkl') for col in label_cols}
one_hot_encoder = joblib.load('/Workspace/Users/pratik.potdar@infocepts.com/mlops/one_hot_encoder.pkl')
scaler = joblib.load('/Workspace/Users/pratik.potdar@infocepts.com/mlops/scaler.pkl')

# COMMAND ----------

label_encoders1= joblib.load('/Workspace/Users/pratik.potdar@infocepts.com/mlops/label_encode.pkl')

# COMMAND ----------

label_encoders

# COMMAND ----------

for columns in label_encoders1:
    cat_cols[columns] = label_encoders1[columns].transform(cat_cols[columns])

# COMMAND ----------

one_hot_cols=['operating_address_city', 'subscription_plan', 'industry_subsector_category']

# COMMAND ----------

one_hot_encoded_incoming = one_hot_encoder.transform(cat_cols[one_hot_cols])
cat_cols = cat_cols.drop(one_hot_cols, axis=1)
cat_cols = pd.concat([cat_cols, pd.DataFrame(one_hot_encoded_incoming)], axis=1)

# COMMAND ----------

log_cols=['total_value_remittance','total_value_cash_deposit','total_num_remittance_deposits','day_since_last_withdrawal','latest_eod_balance','ca_avg_daily_balance_l30d']

# COMMAND ----------

for col in log_cols:
    data[col] = np.log1p(data[col])

# COMMAND ----------

num_cols=data.select_dtypes(include=['float64','int'])


# COMMAND ----------

total=pd.concat([data[list(num_cols.columns)], cat_cols], axis=1)

# COMMAND ----------

total.head()

# COMMAND ----------

#Make Predictions

# COMMAND ----------

total['business_id']=data['business_id']

# COMMAND ----------

target=total[total['had_debcard']==0]

# COMMAND ----------

final=target.drop(['had_debcard','business_id'],axis=1)

# COMMAND ----------

import mlflow

# COMMAND ----------

# Load model from the 'Production' alias
model_uri = f"models:/random_forest_classifier_final@Production"
model = mlflow.sklearn.load_model(model_uri)

# Use the model for inference
#predictions = model.predict(X_test)


# COMMAND ----------

predictions = model.predict(final)

# COMMAND ----------

y_prob=model.predict_proba(final)[:,1]

# COMMAND ----------

target['prediction']=list(predictions)
target['probability']=list(y_prob)

# COMMAND ----------

target.head()

# COMMAND ----------

target['probability'].describe()

# COMMAND ----------


