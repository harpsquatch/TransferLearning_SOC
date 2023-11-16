from dataclasses import dataclass
from pathlib import Path
from typing import List,Dict  # Add this import

@dataclass(frozen=True)   #This is a decorator
class DataIngestionConfig:
    root_dir: Path 
    source_URL: List[Dict[str, str]]
    #local_data_file: Path
    unzip_dir: Path

#The above class takes rootdir, source_url, local_data_file, and unzip_dir as the variables. This we will get from config.yaml and then it will be used in the configuration.py file.

@dataclass(frozen=True)   #This is a decorator
class DataTransformationConfig:
    root_dir: Path 
    training_datasets: str
    data_path: Dict[str, Path]
    train_names: Dict[str, List[str]]
    test_names: Dict[str, List[str]]
    downsampling: bool
    output_capacity: bool
    scale_test: bool
    output_time: bool
    steps: int
    
@dataclass(frozen=True)   #This is a decorator
class ModelTrainerConfig:
    root_dir: Path 
    model_name: str
    steps: int
    num_features: int
    dense_out: int
    num_hidden_units_1: int
    patience: int
    epochs: int
    max_tuner: int
    batch_size: int
    validation_split: int
    numberOfLayers: int
    numberOfLSTMLayers: int
    maxUnits: int
    maxLSTMunits: int
    stepLSTMunit: int
    stepUnit: int
    numberOfDenseLayers: int
    maxDenseUnits: int
    stepDenseUnit: int
    maxDropout: int
    dropoutRateStep: int
    layer: str
    objective_metric: str
    save_dir: Path
    experiment_name: str
    
    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    mlflow_uri: str
    

