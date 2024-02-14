from SOCEst import logger
from SOCEst.pipeline.stage1 import DataIngestionTrainingPipeline
from SOCEst.pipeline.stage2 import DataTransformationTrainingPipeline
from SOCEst.pipeline.stage3 import ModelTrainingPipeline
from SOCEst.pipeline.stage4 import ModelEvaluationTrainingPipeline
import pandas as pd
import os
import numpy as np
from SOCEst.config.configuration import ConfigurationManager


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

for name in range(13,18,1):                  ####################################### THis has to be removed 
    ARTIFACTS_DIR = "artifacts/data_transformation"
    STAGE_NAME = "Data Transformation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        stage2 = DataTransformationTrainingPipeline()
        train_x, train_y, test_x, test_y = stage2.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Model Trainer stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        stage3 = ModelTrainingPipeline(train_x, train_y,name)
        stage3.main()
        experiment_name_tracker = stage3.experiment_name_tracker
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Model evaluation stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        stage4 = ModelEvaluationTrainingPipeline(test_x, test_y,experiment_name_tracker)
        stage4.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
