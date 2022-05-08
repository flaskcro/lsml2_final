import os
import sys
import json
import warnings
from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

import mlflow
import mlflow.sklearn

from data import Data

MLFLOW_SERVER_URL = 'http://127.0.0.1:5000/'

warnings.filterwarnings("ignore")
np.random.seed(40)
data_path = 'data/'

data = Data(data_path)
data.get_data()
data.preprocess_setp1()
data.preprocess_setp2()
data.preprocess_setp3()
data.preprocess_setp4()

X_train, X_valid, y_train, y_valid = data.make_ml_data()

mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
experiment_name = 'future_sales_experiment1'
mlflow.set_experiment(experiment_name)

# run the experiment
with mlflow.start_run():
    alpha = 0.5
    l1_ratio = 0.5

    # model
    params = {'metric': 'rmse',
              'num_leaves': 255,
              'learning_rate': 0.005,
              'feature_fraction': 0.75,
              'bagging_fraction': 0.75,
              'bagging_freq': 5,
              'force_col_wise': True,
              'random_state': 42}

    cat_features = ['shop_id', 'city', 'item_category_id', 'major_category', 'month']
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid)

    # metrics
    lgb_model = lgb.train(params=params,
                          train_set=dtrain,
                          num_boost_round=1500,
                          valid_sets=(dtrain, dvalid),
                          early_stopping_rounds=150,
                          categorical_feature=cat_features,
                          verbose_eval=100)

    predicted_qualities = lgb_model.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, predicted_qualities))
    mae = mean_absolute_error(y_valid, predicted_qualities)
    r2 = r2_score(y_valid, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # save the metric valuse
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(lgb_model, "model")

client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URL)
experiment = client.get_experiment_by_name(experiment_name)
run_info = client.list_run_infos(experiment.experiment_id)[-1]

reg_model_name = 'FinalTask'

client.create_registered_model(reg_model_name)

result = client.create_model_version(
    name=reg_model_name,
    source=f"{run_info.artifact_uri}/model",
    run_id=run_info.run_id
)

client.transition_model_version_stage(
    name=reg_model_name,
    version=result.version,
    stage="Staging"
)

import mlflow.pyfunc
model_name = reg_model_name
stage = 'Staging'

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)

model.predict(X_valid[:10])
client.transition_model_version_stage(
    name=reg_model_name,
    version=result.version,
    stage="Production"
)
os.system('MLFLOW_TRACKING_URI=http://127.0.0.1:5000/ mlflow models serve -m "models:/FinalTask/Staging" -p 5005 --env-manager=local &')

stage = 'Production'
model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{stage}"
)