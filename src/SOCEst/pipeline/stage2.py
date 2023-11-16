from SOCEst.config.configuration import ConfigurationManager
from SOCEst.components.data_transformation import DataTransformation 
from SOCEst import logger
import pandas as pd
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
        
        for dataset_name in data_transformation_config.training_datasets:
            
            cycles = data_transformation.get_discharge_whole_cycle(dataset_name)
            train_x_each, train_y_each, test_x_each, test_y_each = data_transformation.get_discharge_multiple_step(cycles)
            train_y_each = data_transformation.keep_only_y_end(train_y_each)
            test_y_each = data_transformation.keep_only_y_end(test_y_each)
            
            # Append results to the lists
            train_x.append(train_x_each)
            train_y.append(train_y_each)
            test_x.append(test_x_each)
            test_y.append(test_y_each)
            

        logger.info("Training and test sets are prepared and are ready to train")

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

