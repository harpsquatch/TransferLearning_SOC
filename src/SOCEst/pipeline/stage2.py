from SOCEst.config.configuration import ConfigurationManager
from SOCEst.components.data_transformation import DataTransformation 
from SOCEst import logger
import pandas as pd
import numpy as np
import os

STAGE_NAME = "Data Transfromation stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        
        data_transformation_config = config.get_data_transformation_config()
        
        data_transformation = DataTransformation(config=data_transformation_config)
        
        train_x, train_y, test_x, test_y = [], [], [], []
        
        train_y_complete, test_y_complete  = [], []
        
        for dataset_name in data_transformation_config.training_datasets:
            
            cycles = data_transformation.get_discharge_whole_cycle(dataset_name)
            train_x_each, train_y_each, test_x_each, test_y_each = data_transformation.get_discharge_multiple_step(cycles)
            train_y_end = data_transformation.keep_only_y_end(train_y_each)
            test_y_end = data_transformation.keep_only_y_end(test_y_each)
            
            # Append results to the lists
            train_x.append(train_x_each)
            train_y.append(train_y_end)
            test_x.append(test_x_each)
            test_y.append(test_y_end)
            
            train_y_complete.append(train_y_each) #These variables are just for test to see what would happen if we dont implement keep_only_y_end
            test_y_complete.append(test_y_each)
            
        # Concatenate lists of tuples to NumPy arrays
        train_x_arr = np.concatenate(train_x, axis=0)
        train_y_arr = np.concatenate(train_y_complete, axis=0)
        test_x_arr = np.concatenate(test_x, axis=0)
        test_y_arr = np.concatenate(test_y_complete, axis=0)

        # Flatten the arrays
        flat_train_x = train_x_arr.reshape(-1, train_x_arr.shape[-1])
        flat_train_y = train_y_arr.reshape(-1, train_y_arr.shape[-1])
        flat_test_x = test_x_arr.reshape(-1, test_x_arr.shape[-1])
        flat_test_y = test_y_arr.reshape(-1, test_y_arr.shape[-1])

        # Convert flattened arrays to DataFrame
        df_train_x = pd.DataFrame(flat_train_x, columns=["current", "voltage", "temperature"])
        df_train_y = pd.DataFrame(flat_train_y, columns=["soc"])
        df_test_x = pd.DataFrame(flat_test_x, columns=["current", "voltage", "temperature"])
        df_test_y = pd.DataFrame(flat_test_y, columns=["soc"])

        # Save DataFrames to CSV in the artifacts directory
        df_train = pd.concat([df_train_x, df_train_y], axis=1)
        df_test = pd.concat([df_test_x, df_test_y], axis=1)

        df_train.to_csv(os.path.join(data_transformation_config.root_dir, "train_data.csv"), index=False)
        df_test.to_csv(os.path.join(data_transformation_config.root_dir, "test_data.csv"), index=False)

        return train_x, train_y, test_x, test_y

    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        train_x, train_y, test_x, test_y = obj.main()
        #obj.combine_and_save_data(train_x, train_y, test_x, test_y)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

