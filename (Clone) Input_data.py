# Databricks notebook source
# MAGIC %pip install databricks-feature-store

# COMMAND ----------

import pandas as pd
import databricks.feature_store as feature_store
from databricks.feature_store import FeatureStoreClient


# COMMAND ----------

data=pd.read_csv('/Workspace/Users/pratik.potdar@infocepts.com/mlops/mlops_input_data.csv')

# COMMAND ----------

data.drop(['Unnamed: 0'],axis=1,inplace=True)

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

fs.create_table(
    name="Input_Data",   # Name of the feature table
    primary_keys=["business_id"],                       # Primary keys for the table
    df=spark.createDataFrame(data),                                     # The DataFrame to store
    description="The input data to build the propensity model."
)

# COMMAND ----------


