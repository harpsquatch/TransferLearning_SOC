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
        data_pipeline = DataTransformationTrainingPipeline()
        train_x, train_y, test_x, test_y = data_pipeline.main()
        
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        for model_path in model_evaluation_config.model_path:
            model_evaluation_config.log_into_mlflow(model_path,test_x,test_y)
            

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
