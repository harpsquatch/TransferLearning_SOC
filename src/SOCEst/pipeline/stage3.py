from SOCEst.config.configuration import ConfigurationManager
from SOCEst.components.model_trainer import ModelTrainer 
from SOCEst.components.model_trainer import modelHO_New
from SOCEst.components.model_trainer import modelHO_SEGAN_LSTM
from SOCEst.pipeline.stage2 import DataTransformationTrainingPipeline
from SOCEst.components.data_transformation import DataTransformation 
from SOCEst import logger
from tensorflow.keras.models import load_model
import pandas as pd
import h5py
import os


STAGE_NAME = "Model Training Stage"
class ModelTrainingPipeline:
    def __init__(self):
        pass
    def model_training(self): 
        config = ConfigurationManager()
        
        #Get the data from stage3 which is specifically prepared for model training
        data_pipeline = DataTransformationTrainingPipeline()
        train_x, train_y, test_x, test_y = data_pipeline.main()

        #Train and then return the model
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model, history = model_trainer.tune_modelClass(train_x, train_y)
        
        #Save the model
        model.summary()
        model.save(model_trainer.directory + '/models')
        
    def transfer_learning(self):
        config = ConfigurationManager()
        
        #Initialise the data_transformation config
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config) 
        
        #The same configuration for the transfer learning will be used 
        model_trainer_config = config.model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        
        #get training and testing dataset for target_dataset
        cycles = data_transformation.get_discharge_whole_cycle(model_trainer_config.target_dataset )
        train_x, train_y, test_x, test_y = data_transformation.get_discharge_multiple_step(cycles)
        train_y = data_transformation.keep_only_y_end(test_y)
        test_y = data_transformation.keep_only_y_end(test_y)

        #load the model from the model_path provided
        for model_path in model_trainer_config.pretrained_model_path:
            with h5py.File(model_path, 'r') as file:
                model = load_model(file)
                #Implement the transfer learning and return the model
                for tl_technique in model_trainer_config.transfer_learning_technique:
                    tl_model = model_trainer.transfer_learning(train_x, train_y, model, tl_technique)
                    tl_model.summary()
                    #Save the new model
                    tl_model.save(model_trainer.directory + '/models')
        
    
    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.model_trainer_config()
        
        if model_trainer_config.mode == 'model_training':
            self.model_training()
        elif model_trainer_config.mode == 'transfer_learning':
            self.transfer_learning()
        else:
            print("Invalid mode. Choose either 'model_training' or 'transfer_learning'")
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        

