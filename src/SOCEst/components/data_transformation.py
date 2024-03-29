import os
from SOCEst import logger
from pathlib import Path
from SOCEst.entity.config_entity import (DataTransformationConfig)   
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


class DataTransformation: 
    def __init__(self, config: DataTransformationConfig): 
        self.config = config
        
    def get_discharge_whole_cycle(self, dataset):
        train = self._get_data(dataset, self.config.train_names,self.config.downsampling, self.config.output_capacity, self.config.output_time)
        test = self._get_data(dataset, self.config.test_names, self.config.downsampling, self.config.output_capacity, self.config.output_time)
        train, test = self._scale_x(train, test, scale_test=self.config.scale_test)
        return (train, test)
    
    def _get_data(self,dataset,names, downsampling, output_capacity, output_time=False):
            if dataset == "LG":
                names = names.LG
                return self.Lg_get_data(names , downsampling, output_capacity, output_time)
            elif dataset == "Calce":
                names = names.Calce
                return self.Calce_get_data(names , downsampling, output_capacity, output_time)
            elif dataset == "Polimi":
                names = names.Polimi
                return self.Polimi_get_data(names , downsampling, output_capacity, output_time)
            elif dataset == "Madison":
                names = names.Madison
                return self.Madison_get_data(names , downsampling, output_capacity, output_time)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset}")

        
    def Lg_get_data(self, names, downsampling, output_capacity, output_time=False):
        cycles = []
        for name in names:
            cycle = pd.read_csv(self.config.data_path.LG + name + '.csv', skiprows=30)
            print(names)
            cycle.columns = ['Time Stamp','Step','Status','Prog Time','Step Time','Cycle',
                            'Cycle Level','Procedure','Voltage','Current','Temperature','Capacity','WhAccu','Cnt','Empty']
            cycle = cycle[(cycle["Status"] == "TABLE") | (cycle["Status"] == "DCH")]

            max_discharge = abs(min(cycle["Capacity"]))
            cycle["SoC Capacity"] = max_discharge + cycle["Capacity"]
            cycle["SoC Percentage"] = cycle["SoC Capacity"] / max(cycle["SoC Capacity"])
            x = cycle[["Voltage", "Current", "Temperature"]].to_numpy()

            if output_time:
                cycle['Prog Time'] = cycle['Prog Time'].apply(self._time_string_to_seconds)
                cycle['Time in Seconds'] = cycle['Prog Time'] - cycle['Prog Time'][0]

            if output_capacity:
                if output_time:
                    y = cycle[["SoC Capacity", "Time in Seconds"]].to_numpy()
                else:
                    y = cycle[["SoC Capacity"]].to_numpy()
            else:
                if output_time:
                    y = cycle[["SoC Percentage", "Time in Seconds"]].to_numpy()
                else:
                    y = cycle[["SoC Percentage"]].to_numpy()

            if np.isnan(np.min(x)) or np.isnan(np.min(y)):
                logger.info("There is a NaN in cycle " + name + ", removing row")
                x = x[~np.isnan(x).any(axis=1)]
                y = y[~np.isnan(y).any(axis=1)].reshape(-1, y.shape[1])

            if downsampling:
                x = x[0::10]
                y = y[0::10]

            cycles.append((x, y))

        return cycles 
    
    def Calce_get_data(self, names,downsampling ,output_capacity, output_time=False):
        cycles = []
        for name in names:
            cycle = pd.read_csv(self.config.data_path.Calce + name + '.csv')
            print(names)
            x = cycle[["V", "I", "T"]].to_numpy()
            y = cycle[["SOC"]].to_numpy()                  

            if np.isnan(np.min(x)) or np.isnan(np.min(y)):
                self.logger.info("There is a NaN in cycle " + name + ", removing row")
                x = x[~np.isnan(x).any(axis=1)]
                y = y[~np.isnan(y).any(axis=1)].reshape(-1, y.shape[1])

            cycles.append((x, y))

        return cycles
    
    def Polimi_get_data(self,names, downsampling, output_capacity, output_time=False):
        cycles = []

        #for folder in folders:
            # Assuming each folder contains CSV files
            #iles_in_folder = [f for f in os.listdir(self.config.data_path.Polimi + folder) if f.endswith('.csv')]

            #for file in files_in_folder:
                #print(file)
        for name in names:
            cycle = pd.read_csv(self.config.data_path.Polimi + name + '.csv')
            x = cycle[["V","I","T"]].to_numpy()
            y = cycle[["SoC"]].to_numpy()

            # Handle NaN values
            if np.isnan(np.min(x)) or np.isnan(np.min(y)) or (np.max(x) == np.min(x)):
                self.logger.info("There is a NaN in cycle " + name + ", removing row")
                x = x[~np.isnan(x).any(axis=1)]
                y = y[~np.isnan(y).any(axis=1)].reshape(-1, y.shape[1])

            if downsampling:
                # To change the timestep by 10
                x = x[0::10]
                y = y[0::10]

            cycles.append((x, y))

        return cycles
    
    def Madison_get_data(self, names,downsampling ,output_capacity, output_time=False):
        cycles = []
        for name in names:
            cycle = pd.read_csv(self.config.data_path.Madison + name + '.csv')
            print(names)
            cycle.columns = ['Current', 'Voltage', 'Temperature', 'SoC']
            x = cycle[['Current', 'Voltage', 'Temperature']].values
            y = cycle['SoC'].values

            if np.isnan(np.min(x)) or np.isnan(np.min(y)):
                self.logger.info("There is a NaN in cycle " + name + ", removing row")
                x = x[~np.isnan(x).any(axis=1)]
                y = y[~np.isnan(y).any(axis=1)].reshape(-1, y.shape[1])
            
            if downsampling:
                x = x[0::10]
                y = y[0::10]

            y = np.expand_dims(y, axis=1)  # Adds Cycle dimension at the beginning
             # Add dimensions to indicate Cycle, Row, and Column,
             # x = np.expand_dims(x, axis=2)  # Adds Cycle dimension at the beginning
             
            # Rearrange the columns in x to 'Voltage', 'Current', 'Temperature'
            x = x[:, [1, 0, 2]]
            cycles.append((x, y))

        return cycles
    
    
    def _time_string_to_seconds(self, input_string):
        time_parts = input_string.split(':')
        second_parts = time_parts[2].split('.')
        return timedelta(hours=int(time_parts[0]), 
            minutes=int(time_parts[1]), 
            seconds=int(second_parts[0]), 
            microseconds=int(second_parts[1])).total_seconds()

    def _scale_x(self, train, test, scale_test=False):
        for index_feature in range(len(train[0][0][0])):
            feature_min = min([min(cycle[0][:,index_feature]) for cycle in train])
            feature_max = max([max(cycle[0][:,index_feature]) for cycle in train])
            for i in range(len(train)):
                train[i][0][:,index_feature] = (train[i][0][:,index_feature]-feature_min)/(feature_max-feature_min)
            if scale_test:
                for i in range(len(test)):
                    test[i][0][:,index_feature] = (test[i][0][:,index_feature]-feature_min)/(feature_max-feature_min)

        return train, test


    #################################
    #
    # get_stateful_cycle
    #
    #################################
    def get_stateful_cycle(self, cycles, pad_num = 0, steps = 100):
        max_lenght = max(max(len(cycle[0]) for cycle in cycles[0]), max(len(cycle[0]) for cycle in cycles[1]))
        train_x, train_y = self._to_padded_cycle(cycles[0], pad_num, max_lenght)
        test_x, test_y = self._to_padded_cycle(cycles[1], pad_num, max_lenght)
        train_x = self._split_cycle(train_x, steps)
        train_y = self._split_cycle(train_y, steps)
        test_x = self._split_cycle(test_x, steps)
        test_y = self._split_cycle(test_y, steps)
        logger.info("Train x: %s, train y: %s | Test x: %s, test y: %s" %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        return (train_x, train_y, test_x, test_y)

    def _to_padded_cycle(self, cycles, pad_num, max_lenght):
        x_length = len(cycles[0][0][0])
        y_length = len(cycles[0][1][0])
        x = np.full((len(cycles), max_lenght, x_length), pad_num, dtype=float)
        y = np.full((len(cycles), max_lenght, y_length), pad_num, dtype=float)
        for i, cycle in enumerate(cycles):
            x[i, :cycle[0].shape[0]] = cycle[0]
            y[i, :cycle[1].shape[0]] = cycle[1]
        return x, y

    def _split_cycle(self, cycles, steps):
        features = cycles.shape[2]
        time_steps = cycles.shape[1]
        new_cycles = np.empty((0, time_steps//steps, steps, features), float)
        for cycle in cycles:
            new_cycle = np.empty((0, steps, features), float)
            for i in range(0, len(cycle) - steps, steps):
                next_split = np.array(cycle[i:i + steps]).reshape(1, steps, features)
                new_cycle = np.concatenate((new_cycle, next_split))
            new_cycles = np.concatenate((new_cycles, new_cycle.reshape(1, time_steps//steps, steps, features)))
        return new_cycles


    #################################
    #
    # get_discharge_multiple_step
    #
    #################################
    def get_discharge_multiple_step(self, cycles):
        train_x, train_y = self._split_to_multiple_step(cycles[0], self.config.steps)
        test_x, test_y = self._split_to_multiple_step(cycles[1], self.config.steps)
        logger.info("Train x: %s, train y: %s | Test x: %s, test y: %s" %
                         (train_x.shape, train_y.shape, test_x.shape, test_y.shape))
        return (train_x, train_y, test_x, test_y)

    def _split_to_multiple_step(self, cycles, steps):
        x_length = len(cycles[0][0][0])
        y_length = len(cycles[0][1][0])
        x = np.empty((0, steps, x_length), float)
        y = np.empty((0, steps, y_length), float)
        for cycle in cycles:
            for i in range(0, len(cycle[0]) - steps, steps):
                next_x = np.array(cycle[0][i:i + steps]).reshape(1, steps, x_length)
                next_y = np.array(cycle[1][i:i + steps]).reshape(1, steps, y_length)
                x = np.concatenate((x, next_x))
                y = np.concatenate((y, next_y))
        return x, y

    def keep_only_y_end(self, y, is_stateful=False):
        if is_stateful:
            return y[:,:,::self.config.steps]
        else:
            return y[:,::self.config.steps]
    
    def preprocess_and_save_to_csv(self, train_x, train_y, test_x, test_y, data_transformation_config):
    # Concatenate lists of tuples to NumPy arrays
        train_x_arr = np.concatenate(train_x, axis=0)
        train_y_arr = np.concatenate(train_y, axis=0)
        test_x_arr = np.concatenate(test_x, axis=0)
        test_y_arr = np.concatenate(test_y, axis=0)

    # Flatten the arrays
        flat_train_x = train_x_arr.reshape(-1, train_x_arr.shape[-1])
        flat_train_y = train_y_arr.reshape(-1, train_y_arr.shape[-1])
        flat_test_x = test_x_arr.reshape(-1, test_x_arr.shape[-1])
        flat_test_y = test_y_arr.reshape(-1, test_y_arr.shape[-1])

    # Convert flattened arrays to DataFrame
        df_train_x = pd.DataFrame(flat_train_x, columns=["voltage", "current", "temperature"])
        df_train_y = pd.DataFrame(flat_train_y, columns=["soc"])
        df_test_x = pd.DataFrame(flat_test_x, columns=["voltage", "current", "temperature"])
        df_test_y = pd.DataFrame(flat_test_y, columns=["soc"])

    # Save DataFrames to CSV in the artifacts directory
        df_train = pd.concat([df_train_x, df_train_y], axis=1)
        df_test = pd.concat([df_test_x, df_test_y], axis=1)

        df_train.to_csv(os.path.join(data_transformation_config.root_dir, "train_data.csv"), index=False)
        df_test.to_csv(os.path.join(data_transformation_config.root_dir, "test_data.csv"), index=False)

    
