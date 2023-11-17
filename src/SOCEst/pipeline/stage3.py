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
    def model_training(self, config): 
        
        #Get the data from stage3 which is specifically prepared for model training
        data_pipeline = DataTransformationTrainingPipeline()
        train_x, train_y, test_x, test_y = data_pipeline.main()

        model_trainer = ModelTrainer(config=config)
        model, _ = model_trainer.tune_modelClass(train_x, train_y)
        
        #Save the model
        model.summary()
        #model.save(f"{model_trainer.directory}/models/{os.path.basename(model_trainer.directory)}.h5")
        
    def transfer_learning(self,config):
        
        #Get the data from stage3 which is specifically prepared for model training
        data_pipeline = DataTransformationTrainingPipeline()
        train_x, train_y, test_x, test_y = data_pipeline.main()

        model_trainer = ModelTrainer(config=config)
        
        #load the model from the model_path provided
        for model_path in config.pretrained_model_path:
            with h5py.File(model_path, 'r') as file:
                model = load_model(file)
                #Implement the transfer learning and return the model
                for tl_technique in config.transfer_learning_technique:
                    tl_model = model_trainer.transfer_learning(train_x, train_y, model, tl_technique)
                    tl_model.summary()
                    #Save the new model
                    tl_model.save(f"{model_trainer.directory+config.experiment_name}/models/{config.experiment_name}.h5")
                                
    
    def main(self):
        config = ConfigurationManager()
        mode = config.parameters.mode
        model_trainer_config = config.get_model_trainer_config()
        transfer_learning_config = config.get_transfer_learning_config()
        
        if mode == 'model_training':
            self.model_training(model_trainer_config)
        elif mode == 'transfer_learning':
            self.transfer_learning(transfer_learning_config)
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
        

