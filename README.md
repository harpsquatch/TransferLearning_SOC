# Transfer Learning-SOC-Estimation

This repository contains a multi-stage pipeline for estimating the State of Charge (SOC) of a Battery Energy Storage System (BESS) for an electric vehicle. The pipeline includes stages for data ingestion, data transformation, model training, and model evaluation. Trans

## Table of Contents

- [Overview](#overview)
- [Stages](#stages)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Ingestion Stage](#data-ingestion-stage)
  - [Data Transformation Stage](#data-transformation-stage)
  - [Model Training Stage](#model-training-stage)
  - [Model Evaluation Stage](#model-evaluation-stage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

This SOC estimation pipeline leverages various machine learning techniques to accurately estimate the State of Charge (SOC) of an Energy Storage System (ESS). The key stages of the pipeline involve data ingestion, data transformation, model training, and model evaluation. The deployment of transfer learning techniques in the model training stage enables the utilization of pre-trained models on related tasks. In the context of SOC estimation, transfer learning allows the model to leverage knowledge gained from similar energy storage datasets, improving the overall accuracy and efficiency of the SOC estimation process. This approach is particularly valuable when working with limited labelled data, as the pre-trained models bring valuable insights and patterns learned from other relevant domains.

This repository implements a hyperparameter optimization framework for State of Charge (SOC) estimation models, leveraging transfer learning techniques. The project includes custom neural network architectures such as LSTM, BiLSTM, GRU, and their bidirectional variants, combined with Dense layers. It also features advanced tuning capabilities using Keras Tuner (Bayesian and Random Search) and integration with TensorFlow for training on GPU.


## Stages

1. **Data Ingestion Stage:** Downloads and extracts data required for the SOC estimation process.

2. **Data Transformation Stage:** The data processing pipeline in this project handles multiple battery datasets by performing key steps to prepare them for State of Charge (SoC) estimation model training. First, raw datasets from sources such as LG, CALCE, Madison, and Polimi are extracted, followed by data cleaning to handle missing values, noise, and irrelevant features. Outliers are also addressed during this step. Feature engineering focuses on extracting and computing essential SoC-related features, such as voltage, current, and temperature, while aggregating battery behavior over charge/discharge cycles. Once key features are derived, data normalization is applied to ensure consistency, typically using techniques like Min-Max scaling or Z-score normalization. The datasets are then merged to create a large, diverse dataset for transfer learning, ensuring consistent feature spaces and time-series alignment. The processed data is split into training, validation, and testing sets using time-series splitting methods to preserve temporal relationships. Finally, for models like BiLSTM, the data is reshaped into sequences, with sliding windows capturing the sequential nature of battery cycles, grouping consecutive cycles into sequences to model SoC transitions over time.

3. **Model Training Stage:** Utilizes the prepared datasets to train models for SOC estimation, incorporating transfer learning techniques. The Model Training stage supports three modes: training from scratch, transfer learning, and knowledge distillation. In the training from scratch mode, the model is built and trained entirely on the target dataset without leveraging any pre-trained knowledge. In the transfer learning mode, the model initializes with pre-trained weights from a source model, which is fine-tuned on the target dataset to improve performance and reduce training time. Finally, in the knowledge distillation mode, a smaller "student" model is trained to replicate the behavior of a larger "teacher" model, allowing for faster inference while maintaining accuracy.
   Key components:
    **HyperModel Class**: For building SOC estimation models, supporting LSTM, BiLSTM, GRU, and Bidirectional versions.
    **Hyperparameter Optimization**: Uses Bayesian optimization to tune learning rates, activation functions, dropout rates, and network structure.
    **Callback Integration**: EarlyStopping and ModelCheckpoint for efficient model training.
    **Model Trainer**: Combines hyperparameter tuning and transfer learning for optimal model performance.
    **Transfer Learning Module**: Offers 17 different transfer learning techniques for model fine-tuning on new datasets.
    **Knowldege distillation Module**: In this module smaller, more efficient model (student) is trained to mimic the behavior of a larger, more complex model (teacher) to improve inference speed while maintaining performance.
   
5. **Model Evaluation Stage:** Evaluate the trained models using a separate testing dataset. The ModelEvaluation class handles evaluating a machine learning model for State of Charge (SoC) estimation, calculating performance metrics, and logging the results to MLflow. It initializes with a configuration object containing essential parameters like the MLflow URI and file paths. The eval_metrics method computes several key evaluation metrics, including Root Mean Square Percentage Error (RMSPE), Root Mean Square Error (RMSE), Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), Normalized Root Mean Square Error (NRMSE), and FLOPS (Floating Point Operations per Second), though the FLOPS value is currently set to 1 as a placeholder. The log_into_mlflow method loads the trained model, predicts the SoC on the test dataset, calculates the evaluation metrics, and saves both the metrics and the predicted versus actual results locally and in MLflow for experiment tracking.


## Getting Started

### Prerequisites

- Python
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run python main.py

### Configuration and Parameters 
- config.yaml and parameters.yaml files are the two files where you can adjust model and transfer learning configurations as needed.

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Data Sources

The dataset utilized in this SOC estimation pipeline consists of multiple sources, each contributing valuable insights into Energy Storage System (ESS) behavior. If you use this data for any purpose, please make sure to appropriately reference the original sources:

1. **LG Data:**
   - Description: Tests were performed at McMaster University in Hamilton, Ontario, Canada by Dr. Phillip Kollmeyer (phillip.kollmeyer@gmail.com).
   - Reference: https://data.mendeley.com/datasets/cp3473x7xv/3

2. **Madison Data:**
   - Description: Tests were performed at the University of Wisconsin-Madison by Dr. Phillip Kollmeyer (phillip.kollmeyer@gmail.com).
   - Reference: https://data.mendeley.com/datasets/wykht8y7tg/1

3. **NASA Data:**
   - Description: The NASA dataset is obtained from the official NASA website.
   - Reference: https://data.nasa.gov/dataset/Li-ion-Battery-Aging-Datasets/uj5r-zjdb

4. **CALCE Data:**
   - Description: Use of this data for publication purposes should include references to the CALCE article(s) that describe the experiments conducted to generate the data. If you have questions or are interested in contributing your data to the battery data collective, please contact Prof. Michael Pecht.
   - Reference: https://calce.umd.edu/battery-data
  



