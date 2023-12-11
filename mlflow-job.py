# Some imports you will need
import mlflow
import jwt
import json
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

client = mlflow.tracking.MlflowClient()

experiment_name = 'test-experiment'
experiment = client.get_experiment_by_name(name=experiment_name)
if(experiment is None):
    print('Creating experiment ')
    client.create_experiment(name=experiment_name)
    experiment = client.get_experiment_by_name(name=experiment_name)

print(experiment_name)
experiment_id=experiment.experiment_id    
mlflow.set_experiment(experiment_name=experiment_name)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
    
# Read the wine-quality csv file from the URL
csv_url = (
     "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)
try:
     data = pd.read_csv(csv_url, sep=";")
except Exception as e:
     logger.exception(
          "Unable to download training & test CSV, check your internet connection. Error: %s", e
      )

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]


my_log = "This is a test log"
with open("/tmp/test.txt", 'w') as f:
    f.write(my_log)
with open("/tmp/test.log", 'w') as f:
    f.write(my_log)

    
    
#run_tags={'mlflow.user':os.environ['DOMINO_STARTING_USERNAME']}
#Change user name
alpha = 0.7
l1_ratio = 0.6
while(alpha<1):
    with mlflow.start_run():
        print('--- Start Run ---')
        print('Alpha : ' + str(alpha))
        print('L1_Ratio : ' + str(l1_ratio))
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print("mlflow tracking uri:", mlflow.get_tracking_uri())
        print("mlflow tracking type:", tracking_url_type_store)
        
        mlflow.log_artifact("/tmp/test.txt")
        alpha=alpha+0.1
        l1_ratio = l1_ratio + 0.05
        print("--- End Run ---")