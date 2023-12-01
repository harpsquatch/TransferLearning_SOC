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
from keras_flops import get_flops
import h5py
import math
from tensorflow.keras.models import load_model

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, model, actual, pred):
        rmse = 100 * math.sqrt(np.mean(np.square(np.subtract(pred, actual)))) / np.mean(actual)
        mse = 100 * np.mean(np.mean(np.square(np.subtract(pred, actual))))
        mae = 100 * np.mean(np.abs(np.subtract(pred, actual)))
        # Replace with actual function to get FLOPS
        flops = get_flops(model, batch_size=64)

        return rmse, mse, mae, flops

    def log_into_mlflow(self, model_path, experiment_name, test_x, test_y):
        with h5py.File(model_path, 'r') as file:
            model = load_model(file)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_SOC = model.predict(test_x)

            (rmse, mse, mae, flops) = self.eval_metrics(model, test_y, predicted_SOC)
            
            test_y = np.asarray(test_y)
            predicted_SOC = np.asarray(predicted_SOC)

            # Saving predicted vs actual figures
            results_df = pd.DataFrame({'Actual': test_y.flatten(), 'Predicted': predicted_SOC.flatten()})
            results_df.to_csv(os.path.join(self.config.metric_file_name, f"{experiment_name}_results.csv"), index=False)

            # Saving metrics as local
            scores = {"rmse": rmse, "mse": mse, "mae": mae, "flops": flops}
            save_json(path=Path(os.path.join(self.config.metric_file_name, f"{experiment_name}.json")), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("flops", flops)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.keras.log_model(model, "model", registered_model_name="Transfer Learning")
            else:
                mlflow.keras.log_model(model, "model")
