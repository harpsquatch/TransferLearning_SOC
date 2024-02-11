import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.keras
import numpy as np
import joblib
from SOCEst.entity.config_entity import ModelEvaluationConfig
from SOCEst.utils.common import save_json
from pathlib import Path
#from keras_flops import get_flops
import h5py
import math
from tensorflow.keras.models import load_model

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, model, actual, pred):
        rmspe = (np.sqrt(np.mean(np.square(np.subtract(actual, pred)/ actual)))) * 100
        mse = np.square(np.subtract(actual,pred)).mean() 
        nrmse = math.sqrt(np.mean(np.square(np.subtract(pred, actual))))/np.mean(actual)
        rmse = math.sqrt(mse)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        mae = np.mean(np.abs(actual - pred)) 
        # Replace with actual function to get FLOPS
        #flops = get_flops(model, batch_size=64)
        flops = 1

        return rmspe, rmse, mse, mape,mae, flops , nrmse

    def log_into_mlflow(self, model_path, experiment_name, test_x, test_y):
        with h5py.File(model_path, 'r') as file:
            model = load_model(file)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_SOC = model.predict(test_x)

            #(rmspe, rmse, mse, mape, flops) = self.eval_metrics(model, test_y, predicted_SOC)
            
            test_y = np.asarray(test_y)
            predicted_SOC = np.asarray(predicted_SOC)

            # Saving predicted vs actual figures
            results_df = pd.DataFrame({'Actual': test_y.flatten(), 'Predicted': predicted_SOC.flatten()})
            results_df.to_csv(os.path.join(self.config.metric_file_name, f"{experiment_name}_results.csv"), index=False)
            (rmspe, rmse, mse, mape,mae, flops,nrmse) = self.eval_metrics(model, results_df['Actual'], results_df['Predicted'])
            # Saving metrics as local
            scores = {"rmspe": rmspe,"rmse": rmse, "mse": mse, "mape": mape, "mae": mae, "flops": flops, "nrmse":nrmse}
            save_json(path=Path(os.path.join(self.config.metric_file_name, f"{experiment_name}.json")), data=scores)

#            mlflow.log_params(self.config.all_params)
#            mlflow.log_metric("rmspe", rmspe)
#            mlflow.log_metric("rmse", rmse)
#            mlflow.log_metric("mse", mse)
#            mlflow.log_metric("mape", mape)
#            mlflow.log_metric("mae", mape)
#            mlflow.log_metric("flops", flops)
#            mlflow.log_metric("nrmse", nrmse)

            # Model registry does not work with file store
#            if tracking_url_type_store != "file":
                # Register the model
#                mlflow.keras.log_model(model, "model", registered_model_name="Transfer Learning")
#            else:
#                mlflow.keras.log_model(model, "model")
