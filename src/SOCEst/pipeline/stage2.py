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
        
        data_transformation.preprocess_and_save_to_csv(train_x,train_y_complete,test_x,test_y_complete,data_transformation_config)

   
        # Concatenate the data if multiple datasets are provided
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
        
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

