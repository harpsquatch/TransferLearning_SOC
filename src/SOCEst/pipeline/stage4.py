from SOCEst.config.configuration import ConfigurationManager
from SOCEst.components.model_evaluation import ModelEvaluation
from SOCEst.pipeline.stage2 import DataTransformationTrainingPipeline
from SOCEst import logger
import os 

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self,test_x, test_y,experiment_name_tracker):
        self.test_x = test_x
        self.test_y = test_y
        self.experiment_name_tracker = experiment_name_tracker
        pass

    def main(self):
        config = ConfigurationManager()
        
        mode = config.parameters.mode
        model_trainer_config = config.get_model_trainer_config()
        transfer_learning_config = config.get_transfer_learning_config()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
        if mode == 'model_training':
            config = model_trainer_config
        elif mode == 'transfer_learning':
            config = transfer_learning_config
        else:
            print("Invalid mode. Choose either 'model_training' or 'transfer_learning'")
        
        
        for experiment_name in self.experiment_name_tracker:
            experiment_path = os.path.join(model_evaluation_config.model_path, experiment_name, f"{experiment_name}.h5")
            
            model_evaluation.log_into_mlflow(experiment_path,experiment_name,self.test_x,self.test_y)
            

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        config = ConfigurationManager()
        stage2 = DataTransformationTrainingPipeline()
        train_x, train_y, test_x, test_y = stage2.main()
        model_evaluation_config = config.get_model_evaluation_config()
        experiment_name = 'test7'
        for path in model_evaluation_config.specific_model_path: 
            experiment_path = path
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow(experiment_path,experiment_name,test_x,test_y)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
