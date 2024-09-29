# Databricks notebook source
import databricks.feature_store as feature_store
from databricks.feature_store import FeatureStoreClient
import pandas as pd
import numpy as np

# COMMAND ----------

fs = FeatureStoreClient()

# COMMAND ----------

df = fs.read_table(name='innovationday2024ml.default.preprocessing')

# COMMAND ----------

data=df.toPandas()

# COMMAND ----------

data.head().round(1)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# COMMAND ----------

scaler = StandardScaler()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

x=data.drop(['business_id','had_debcard'],axis=1)

# COMMAND ----------

scaler.fit(x)
x=scaler.transform(x)

# COMMAND ----------

y=data['had_debcard']

# COMMAND ----------



# COMMAND ----------

import mlflow
import mlflow.sklearn  # Or use `mlflow.spark` for Spark models
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model


# COMMAND ----------

#mlflow.set_experiment("/Workspace/Users/pratik.potdar@infocepts.com/mlops/mlops_exp1")


# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# COMMAND ----------

    # Example model: Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

# COMMAND ----------

signature = infer_signature(X_test, y_pred)

# COMMAND ----------

experiment_name = "/Users/pratik.potdar@infocepts.com/mlops/Model_training_final"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "random_forest_model_final", signature=signature)

    # Register the model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model_final"
    model_version = mlflow.register_model(model_uri, "random_forest_classifier_final")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')  # Use 'macro' for multi-class
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    
    # Log hyperparameters (optional)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", None)  



print(f"Model registered with version {model_version.version}")

# COMMAND ----------

client = MlflowClient()

# Get the registered model details
model_name = "random_forest_classifier_final"
model_version = model_version.version

# COMMAND ----------



# COMMAND ----------

from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Assign 'Staging' alias to the model version
client.set_registered_model_alias(
    name="random_forest_classifier_final",  # Model name
    alias="Staging",  # Alias you want to assign
    version=model_version    #model_version.version  # Model version you want to point to Staging
)

print(f"Model version {model_version} assigned to 'Staging' alias")

# Similarly, you can assign 'Production' alias
client.set_registered_model_alias(
    name="random_forest_classifier_final",
    alias="Production",
    version=model_version  #model_version.version  # Specify the appropriate version here
)

#print(f"Model version {model_version.version} assigned to 'Production' alias")


# COMMAND ----------

# Change the 'Production' alias to point to a different version
client.set_registered_model_alias(
    name="random_forest_classifier_final",
    alias="Production",
    version=model_version  # Point to the new version
)

print(f"Model version {model_version} assigned to 'Production' alias")


# COMMAND ----------

# Load model from the 'Production' alias
model_uri = f"models:/random_forest_classifier_final@Production"
model = mlflow.sklearn.load_model(model_uri)

# Use the model for inference
#predictions = model.predict(X_test)


# COMMAND ----------

#predictions = model.predict(X_test)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------


