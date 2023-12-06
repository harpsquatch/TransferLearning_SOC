from SOCEst.constants import *                                  #Constants stores filepaths of yaml files we have in the directory 
from SOCEst.utils.common import read_yaml, create_directories   #From common, common functionalities are being imported 
from SOCEst.entity.config_entity import DataIngestionConfig     #Dataingestion configuration is being imported from config file where the class structure has been defined. 
from SOCEst.entity.config_entity import DataTransformationConfig
from SOCEst.entity.config_entity import ModelTrainerConfig
from SOCEst.entity.config_entity import ModelEvaluationConfig
from SOCEst.entity.config_entity import TransferLearningConfig
from box import ConfigBox
from box import Box, BoxList
from datetime import datetime




# This class will be responsible for managing configuration files. It reads config.yaml and creates necessary necessary directories in the artifacts folder  
class ConfigurationManager:  
    
    #The constructor take yaml file paths as arguments. 
    def __init__(self, config_filepath = CONFIG_FILE_PATH, #Config.yaml path 
                       params_filepath = PARAMS_FILE_PATH, #Params.yaml path 
              
                ): 

        self.config = read_yaml(config_filepath) #Config.yaml file is being read 
        self.parameters = read_yaml(params_filepath) #Params.yaml file is being read 


        create_directories([self.config.artifacts_root]) #artifacts_root = artifacts so new folder artifact is created with this

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion  #take the configuration

        create_directories([config.root_dir]) #Create the root_dir = artifacts/data_ingestion

        data_ingestion_config = DataIngestionConfig(     #Here the data ingestion class is being inititalised by passing the neccessary variables. 
            root_dir=config.root_dir,                    
            source_URL=config.source_URL,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config                     
        #Now from yaml file, the parameters data_ingestion class is successfully created. Next this has to be used in src/data_ingestion
    
    def get_data_transformation_config(self) -> DataTransformationConfig: 
        config = self.config.data_transformation
        params = self.parameters.data_parameters
        mode = self.parameters.mode
        model_training_datasets = params.training_datasets
        target_dataset = self.parameters.transfer_learning_parameters.target_dataset
        
        create_directories([config.root_dir]) #Create the root_dir = artifacts/data_transformation 

        if mode == "model_training":
            training_datasets = model_training_datasets

        elif mode == "transfer_learning":
            training_datasets = target_dataset
            
        else: 
            print("Incorrect mode ")
        
        #Filter out the train_names_dictionary based on what is given in training datasets
        filtered_dictionary = {dataset: config.train_names_dictionary.get(dataset, f"{dataset} not found in train_names") for dataset in training_datasets}

        #With the following we can call config.train_names.LG to get the respective training names
        train_names = ConfigBox(filtered_dictionary)
        
        #Filter out the test_names_dictionary based on what is given in training datasets
        filtered_dictionary = {dataset: config.test_names_dictionary.get(dataset, f"{dataset} not found in train_names") for dataset in training_datasets}
        
        #With the following we can call config.test_names.LG to get the respective training names
        test_names = ConfigBox(filtered_dictionary)

        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            training_datasets = training_datasets,     
            data_path=config.data_path,
            train_names = train_names,
            test_names = test_names,
            downsampling = params.downsampling,
            output_capacity = params.output_capacity,
            scale_test = params.scale_test,
            output_time = params.output_time,
            steps = params.steps            
        )
        
        return data_transformation_config

    
    def get_model_trainer_config(self) -> ModelTrainerConfig: 
        config = self.config.model_trainer
        mode = self.parameters.mode
        params = self.parameters.model_parameters
        experiment_name = f"{''.join(self.parameters.data_parameters.training_datasets)}_{datetime.now().strftime('%H%M')}"
                
        create_directories([config.root_dir]) #Create the root_dir = artifacts/data_transformation 
                
        model_trainer_config = ModelTrainerConfig(
            input_dim=config.input_dim, 
            root_dir=config.root_dir,     
            steps = params.steps,
            num_features = params.num_features,
            dense_out = params.dense_out,
            num_hidden_units_1 = params.num_hidden_units_1,
            patience = params.patience,
            epochs = params.epochs,
            max_tuner = params.max_tuner,
            batch_size = params.batch_size,
            validation_split = params.validation_split,
            numberOfLayers = params.numberOfLayers,
            numberOfLSTMLayers = params.numberOfLSTMLayers,
            maxUnits = params.maxUnits,
            maxLSTMunits = params.maxLSTMunits,
            stepLSTMunit = params.stepLSTMunit,
            stepUnit = params.stepUnit,     
            numberOfDenseLayers = params.numberOfDenseLayers,
            maxDenseUnits = params.maxDenseUnits,
            stepDenseUnit = params.stepDenseUnit,
            maxDropout = params.maxDropout,
            dropoutRateStep = params.dropoutRateStep,
            layer = params.layer, 
            objective_metric = params.objective_metric, 
            experiment_name = experiment_name,
            experiment_name_tracker =  config.experiment_name_tracker
        )
        
        return model_trainer_config
    
    def get_transfer_learning_config(self) -> TransferLearningConfig: 
        config = self.config.model_trainer
        params = self.parameters.transfer_learning_parameters
        training_datasets = self.parameters.data_parameters.training_datasets
        target_dataset = self.parameters.transfer_learning_parameters.target_dataset
        
        experiment_name = f"{'_'.join(target_dataset)}_TL{params.transfer_learning_technique}{datetime.now().strftime('%H%M')}"

        #The following is just transfer learning, Get the path for the pretrained paths
        filtered_dictionary = {model: config.pretrained_model_path_dictionary.get(model, f"{model} not found in train_names") for model in params.pretrained_model}
        box_filtered_dictionary = ConfigBox(filtered_dictionary)
        
        pretrained_model_path = [ ]
        for model_name in params.pretrained_model:
            path = getattr(box_filtered_dictionary, model_name, f"{model_name} not found in box_filtered_dictionary")
            pretrained_model_path.append(path)
        
                
        transfer_learning_config = TransferLearningConfig(
            patience = params.patience,
            root_dir = config.root_dir,
            epochs = params.epochs,
            batch_size = params.batch_size,
            validation_split = params.validation_split,
            layer = params.layer, 
            pretrained_model_path = pretrained_model_path,
            experiment_name = experiment_name, 
            target_dataset = params.target_dataset,
            pretrained_model = params.pretrained_model,
            transfer_learning_technique = params.transfer_learning_technique,
            experiment_name_tracker =  config.experiment_name_tracker
        )
        
        return transfer_learning_config
    
    
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation   
        params = self.parameters.model_parameters
        create_directories([config.root_dir])
        
        #In case you want to pass the model_name and then evaluate 
        #filtered_dictionary = {model: config.model_path_dictionary.get(model, f"{model} not found in train_names") for model in config.model_for_evaluation}
        #box_filtered_dictionary = ConfigBox(filtered_dictionary)
        
        #model_path = [ ]
        #for model_name in config.model_for_evaluation:
        #   path = getattr(box_filtered_dictionary, model_name, f"{model_name} not found in box_filtered_dictionary")
        #    model_path.append(path)

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            mlflow_uri="https://dagshub.com/harpreets924/TransferLearning_SOC.mlflow",
        )

        return model_evaluation_config
