from SOCEst.config.configuration import ConfigurationManager
from SOCEst.components.model_evaluation import ModelEvaluation
from SOCEst.pipeline.stage2 import DataTransformationTrainingPipeline
from SOCEst import logger

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.model_trainer_config()
        
        if model_trainer_config.mode == 'model_training':
            data_pipeline = DataTransformationTrainingPipeline()
            train_x, train_y, test_x, test_y = data_pipeline.main()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
            model_evaluation_config.log_into_mlflow(model_path,test_x,test_y)
            
        elif model_trainer_config.mode == 'transfer_learning':
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



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
