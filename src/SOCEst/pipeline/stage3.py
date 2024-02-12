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
    def __init__(self,train_x, train_y,technique): #technique
        self.train_x = train_x
        self.train_y = train_y
        self.technique = technique        ###################################### THis has to be removed 
    def model_training(self, config): 
        
        #Get the data from stage3 which is specifically prepared for model training
        #data_pipeline = DataTransformationTrainingPipeline()
        #train_x, train_y, test_x, test_y = data_pipeline.main()

        model_trainer = ModelTrainer(config=config)
        model, _ = model_trainer.tune_modelClass(self.train_x, self.train_y)
        
        #Save the model
        model.summary()
        #model.save(f"{model_trainer.directory}/models/{os.path.basename(model_trainer.directory)}.h5")
        config.experiment_name_tracker.append(model_trainer.config.experiment_name)
        print(f'experiment_name_tracker:{config.experiment_name_tracker}')
        
    def transfer_learning(self,config):
        
        #Get the data from stage3 which is specifically prepared for model training
        #data_pipeline = DataTransformationTrainingPipeline()
        #train_x, train_y, test_x, test_y = data_pipeline.main()

        model_trainer = ModelTrainer(config=config)
        
        #load the model from the model_path provided
        print("pretrained_model_path",config.pretrained_model_path)
        
        for model_path in config.pretrained_model_path:
            with h5py.File(model_path, 'r') as file:
                #Load the model 
                model = load_model(file, compile=False)
                
                model.summary()
                #Collect the name of the model for documentation 
                base_filename = os.path.splitext(os.path.basename(model_path))[0]
                
                #Implement the transfer learning and return the model
                #config.transfer_learning_technique = self.technique

                #for tl_technique in config.transfer_learning_technique:
                #tl_model = model_trainer.transfer_learning(self.train_x, self.train_y, model, config.transfer_learning_technique)
                tl_model = model_trainer.transfer_learning(self.train_x, self.train_y, model, self.technique)            ####################################### transfer_learning_technique has to be added 
                tl_model.summary()
                
                #New experiment name is created in order to correctly distingush the models
                #new_experiment_name = f"{base_filename}_{config.experiment_name}_TL_technique{config.transfer_learning_technique}"
                
                new_experiment_name = f"{base_filename}_{config.experiment_name}_TL_technique{self.technique}"

                #Save the model 
                tl_model.save(f"{config.root_dir}/{new_experiment_name}/{new_experiment_name}.h5")
                
                #experiment name tracker, appends all the ecperiment inside the list and then same list is used while evaluating the models
                config.experiment_name_tracker.append(new_experiment_name)

       
    
    def main(self):
        config = ConfigurationManager()
        mode = config.parameters.mode
        model_trainer_config = config.get_model_trainer_config()
        transfer_learning_config = config.get_transfer_learning_config()
        
        if mode == 'model_training':
            self.model_training(model_trainer_config)
            self.experiment_name_tracker = model_trainer_config.experiment_name_tracker
        elif mode == 'transfer_learning':
            self.transfer_learning(transfer_learning_config)
            self.experiment_name_tracker = transfer_learning_config.experiment_name_tracker
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
        

