# Transfer Learning-SOC-Estimatio

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

## Stages

1. **Data Ingestion Stage:** Downloads and extracts data required for the SOC estimation process.

2. **Data Transformation Stage:** Prepares training and testing datasets for model training.

3. **Model Training Stage:** Utilizes the prepared datasets to train models for SOC estimation, incorporating transfer learning techniques.

4. **Model Evaluation Stage:** Evaluate the trained models using a separate testing dataset.


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
  



