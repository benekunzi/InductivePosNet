import pandas as pd
import numpy as np
import os 
import glob
from typing import List, Tuple
import pickle
import warnings
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, BatchNormalization, Add, Dropout, LSTM
from tensorflow.keras.optimizers.schedules import CosineDecay
# try:
#     from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay, CosineDecay
# except ImportError:
#     from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from keras.callbacks import CSVLogger
from keras.initializers import HeUniform
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.regularizers import L1, L2, L1L2
import metrics
import datetime
import time
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import argparse

# um in den server zu kommen
# vpn 
# ssh gpu18
# tmux attach -t 0
# interactive (bleibt für 24h)
# bash baseline.sh

class RegressionModel():
    def __init__(self) -> None:
        self.directory_path =  '01_measurement_data'
        self.model = Sequential()
        self.config = wandb.config
        self.validation_split = 0.3
        self.center_distance = 100
        self.y_max = 500
        self.y_min = -500
        self.x_max = 600
        self.x_min = -600 
        self.start_points = []
        self.end_points = []
        self.merged_df_list = []
        self.x_coordinates = []
        self.y_coordinates = []
    
    def __sort_key(self, file_path) -> None:
        """sorting key function to sort filenames alphabetically"""
        
        file_name = os.path.basename(file_path)
        
        return file_name

    def __get_csv_paths_list(self, directory_path: str) ->  List[str]:
        """ this function is meant to list all A4T-, A5T-, A6T- and A7T-measurement files
            for different coil-coil-distances in form of her absolute filepathes."""

        csv_files = []

        parPath = os.getcwd()
        dataFolder = directory_path
        dataPath = os.path.join(parPath, dataFolder)

        subfolders = [f.path for f in os.scandir(dataPath) if f.is_dir()]

        for folder in subfolders:
            files_in_dir = glob.glob(os.path.join(folder, "*.csv"))
            sorted_files = sorted(files_in_dir, key= self.__sort_key)
            csv_files.append(sorted_files)

        csv_files = csv_files[0:-1]

        return csv_files
    
    def __extend_array(self, arr: np.ndarray, n: int):
        if arr.ndim == 1:
            result = []
            result.extend([arr[0]])
            for i in range(1, len(arr)):
                diff = (arr[i] - arr[i-1]) / (n+1)
                new_elements = [arr[i-1] + j*diff for j in range(1, (n+1))]
                result.extend(new_elements)
                result.extend([arr[i]])
            return np.array(result)

        result = []
        for row in arr:
            new_row = [row[0]]
            for i in range(1, len(row)):
                diff = (row[i] - row[i-1]) / (n+1)
                new_elements = [row[i-1] + j*diff for j in range(1, (n+1))]
                new_row.extend(new_elements)
                new_row.append(row[i])
            result.append(new_row)
        return np.array(result)
    
    def __process_csv_files(self, csv_block: List[str]):
        """ notes:
            parameter csv_block needs to be a list of 4 filepath strings.
            those strings refer to 4 different measurement files of sensors
            A4T, A5T, A6T and A7T.
            for a different height (coil-coil-distance) another csv_block
            has to be passed in.
            this function returns a combined/merged pd.dataframe in which
            the structure is ["A4T","A5T","A6T","A7T","x","y","z"], 
            thus all XxY-grids within each individual measurment file
            are flattened."""

        tmp_dfs = []
        columns = ["A4T","A5T","A6T","A7T","x","y","z"]
        merged_df = pd.DataFrame(columns=columns)

        curr_dz = float(csv_block[0].split("_dz")[1].split("/")[0]) # extract current z-height of dataset
        for i, sensor_file in enumerate(csv_block):
            data_list = [] # create a list to store the data

            curr_sensor = sensor_file.split("_")[-1].split(".")[0]  # extract current sensor's name in dataset

            df = pd.read_csv(sensor_file,skiprows=28,sep=",",index_col=0)
            df.index = pd.to_numeric(df.index)
            df.columns = pd.to_numeric(df.columns)

            tmp_dfs.append(df)
            
        return tmp_dfs
    
    def __linear_interpolation(self, start, end, number_of_steps=50):
        linear_func = []

        if start[0] != end[0]:
            xs = np.linspace(start[0], end[0], number_of_steps)
            ys = np.linspace(start[1], end[1], number_of_steps)

            for i in range(xs.size):
                linear_func.append([xs[i], ys[i]])

        else:
            ys = np.linspace(start[1], end[1], number_of_steps)
            xs = []

            for i in range(ys.size):
                xs.append(start[0])
            
            for i in range(ys.size):
                linear_func.append([xs[i], ys[i]])

        return linear_func
    
    def __find_nearest_coordinates(self, coordinates, grid):
        squared_distances = np.sum((grid - coordinates)**2, axis=1)
        min_index = np.argmin(squared_distances)

        return grid[min_index]
    
    def __generate_start_points(self) -> None:
        for x in self.x_coordinates:
            if x <= self.x_max and x >= self.x_min:
                self.start_points.append([x, self.y_max])

    def __generate_end_points(self) -> None:
        for x in self.x_coordinates:
            if x <= self.x_max and x >= self.x_min:
                self.end_points.append([x, self.y_min])

    def __generate_linear_functions(self) -> np.ndarray:
        linear_functions = []
        for i in range(len(self.start_points)):
            for j in range(len(self.end_points)):
                linear_functions.append(self.__linear_interpolation(self.start_points[i], self.end_points[j]))

        return np.array(linear_functions)

    def __generate_linear_functions_rv(self, linear_functions: np.ndarray):
        x_grid, y_grid = np.meshgrid(self.x_coordinates, self.y_coordinates)
        coordinate_grid = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

        linear_functions_real = []
        for linear in linear_functions:
            temp_list = [self.__find_nearest_coordinates(coord, coordinate_grid) for coord in linear]
            linear_functions_real.append(temp_list)

        return np.array(linear_functions_real)

    def __generate_voltage_values(self, linear_functions_rv: np.ndarray) -> np.ndarray:
        voltage_values = []

        for height in self.merged_df_list:
            for linear in linear_functions_rv:
                temp_list = []
                for coord in linear:
                    temp_list.append([height[i].at[coord[0], coord[1]] for i in range(len(self.merged_df_list[0]))])
                voltage_values.append(temp_list)

        return np.array(voltage_values)
    
    def __generate_coordinates(self, linear_functions_rv: np.ndarray):
        temp_linear_real = linear_functions_rv
        coordinate_values = []

        for i in range(0, len(self.merged_df_list)):
            for linear in temp_linear_real:
                coordinate_values.append(linear[linear.shape[0]-1])

        return np.array(coordinate_values)

    def __generate_all_paths(self, scaled_input_data: np.ndarray, scaled_target_data: np.ndarray):
        array_length = scaled_input_data.shape[0]

        scaled_input_temp_full = []
        scaled_target_temp_full = []

        for i in range(0, array_length):
            linear_length = scaled_input_data[i].shape[0]
            temp_input = []
            for j in range(0, linear_length):
                scaled_target_temp_full.append(scaled_target_data[i])

                temp_input.append(scaled_input_data[i][j])

                scaled_input_temp = temp_input

                for k in range(0, scaled_input_data[i].shape[0] - j - 1):
                    scaled_input_temp = np.vstack((np.array([2, 2, 2, 2]), scaled_input_temp))

                scaled_input_temp_full.append(scaled_input_temp)

        scaled_input_temp_full = np.array(scaled_input_temp_full)
        scaled_target_temp_full = np.array(scaled_target_temp_full)

        print(scaled_input_temp_full.shape, scaled_target_temp_full.shape)

        return (scaled_input_temp_full, scaled_target_temp_full)


    def generate_list(self) -> List: 
        csv_files = []
        merged_df_list = []

        csv_files = self.__get_csv_paths_list(self.directory_path) # function call to get the filepathes of all measurement files
        csv_files # jupyter print

        for idx, z_height_group in enumerate(csv_files): # loop through each group of files (A4T, A5T, A6T, A7T)
            merged_df_list.append(self.__process_csv_files(z_height_group))

        return merged_df_list

    def generate_inputData(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generates input data"""
        return dataframe[["A4T", "A5T", "A6T", "A7T"]]
    
    def generate_targetData(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generates target data"""
        return dataframe[["x", "y"]]
    
    def load_scaler(self, path_scaler: str):
        with open(path_scaler, 'rb') as file:
            scaler = pickle.load(file)

        return scaler

    def scale(self, scaler, data: pd.DataFrame) -> np.ndarray:
        return scaler.transform(data)
    
    def train_test_split(self, features: np.ndarray, target: np.ndarray) -> list:
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.validation_split, random_state=42)

        return (X_train, X_test, y_train, y_test)

    def generate_weights(self, scaled_target_data: np.ndarray) -> np.ndarray:
        diff_from_1 = np.linalg.norm([-0.5, 0.0] - scaled_target_data, axis=1)
        output_weights = np.exp(-(diff_from_1/0.5)**2)

        return output_weights

    def createScheduler(self, size: int, inital_learning_rate: float, epochs: int) -> CosineDecay:
        print("decay steps:", np.ceil((size * (1 - self.validation_split)) / self.config.batch_size) * epochs)
        print(size, self.config.batch_size, epochs)
        return CosineDecay(
            initial_learning_rate = inital_learning_rate,
            decay_steps = np.ceil((size * (1 - self.validation_split)) / self.config.batch_size) * epochs,
        )
        # return ExponentialDecay(
        #     initial_learning_rate= self.config.learning_rate,
        #     decay_steps= (size // self.config.batch_size),
        #     decay_rate= self.config.learning_rate_decay
        # )

    def buildModel(self, shape):
        n_features = 4

        self.model = Sequential()
        self.model.add(Input(shape=shape, name='input'))
        self.model.add(LSTM(128, return_sequences=False, name='lstm_hidden'))
        self.model.add(Dense(self.config.dense_neurons, name='Dense_hidden'))
        self.model.add(Activation(self.config.activation_hidden, name='activation_hidden'))
        self.model.add(Dense(2, name='Dense_Out'))
        self.model.add(Activation(self.config.activation_final, name='activation_Out'))

    def compileModel(self, prescaler_target_data: MinMaxScaler, lr_scheduler):
        optimizer = Adam(learning_rate= lr_scheduler)
        loss = MeanSquaredError()

        self.model.compile(
            optimizer=optimizer, 
            loss=loss,
            metrics=[
                "mse",
                metrics.CenterMeanXYNormRealWorld(scaler=prescaler_target_data),
                metrics.MeanXYNormRealWorld(scaler=prescaler_target_data),
                metrics.MeanXYSquarredErrorRealWorld(scaler=prescaler_target_data),  
                metrics.MeanXYAbsoluteErrorRealWorld(scaler=prescaler_target_data),
                metrics.MeanXYRootSquarredErrorRealWorld(scaler=prescaler_target_data),
                metrics.CenterMaxXYNormRealWorld(scaler=prescaler_target_data),
                # metrics.CenterWithinTresholdXYNormRealWorld(scaler=prescaler_target_data)
            ],
            weighted_metrics= []
        )

    def trainModel(self, scaled_input_data: np.ndarray, scaled_target_data: np.ndarray, validation_data) -> None:
        timestamp = time.time()
        datetime_obj = datetime.datetime.fromtimestamp(timestamp)
        # Format the datetime object as a string in your desired format
        readable_time = datetime_obj.strftime('%Y-%m-%d%H:%M:%S')

        identifier = f'{wandb.run.id}' # wandb creates id for model and save it online

        mcp_save = ModelCheckpoint(f'trained_models/{identifier}.h5', 
                        save_best_only=True,
                        monitor='val_loss', 
                        mode='min')
        
        csv_logger = CSVLogger(f'logging/{identifier}.csv',
                                append=False,
                                separator=',')
        callbacks = [
            csv_logger,
            WandbMetricsLogger(log_freq=100),
            mcp_save
        ]

        history = self.model.fit(
            x = scaled_input_data,
            y = scaled_target_data, 
            epochs = self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks = callbacks,
            shuffle=True,
            validation_data= validation_data
        )

        # with open(f'history/{identifier}.pkl', 'wb') as file:
        #     pickle.dump(history, file)

    def main(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        self.merged_df_list = self.generate_list()

        self.x_coordinates =  self.merged_df_list[0][0].columns.astype(int).to_numpy()
        self.y_coordinates =  self.merged_df_list[0][0].index.astype(int).to_numpy()

        print(self.x_coordinates)
        print(self.y_coordinates)

        self.__generate_start_points()
        self.__generate_end_points()

        print(self.start_points[0])
        print(self.end_points[0])

        linear_functions = self.__generate_linear_functions()

        print(len(linear_functions))

        linear_functions_rv = self.__generate_linear_functions_rv(linear_functions)

        print(linear_functions_rv.shape)

        voltage_values = self.__generate_voltage_values(linear_functions_rv)

        coordinate_values = self.__generate_coordinates(linear_functions_rv)

        scaler_input = self.load_scaler("prescaler_input_data_Range-1-1.pkl")
        scaler_target = self.load_scaler("prescaler_target_data_Range-1-1.pkl")

        scaled_input_data = []

        for i in range(voltage_values.shape[0]):
            scaled_input_data.append(scaler_input.transform(voltage_values[i]))
        scaled_input_data = np.array(scaled_input_data)

        scaled_target_data = scaler_target.transform(coordinate_values)
        scaled_target_data = np.array(scaled_target_data)

        scaled_input_data, scaled_target_data = self.__generate_all_paths(scaled_input_data, scaled_target_data)

        X_train, X_test, y_train, y_test = self.train_test_split(scaled_input_data, scaled_target_data)
        
        lr_scheduler = self.createScheduler(X_train.shape[0], inital_learning_rate=self.config.learning_rate, epochs=self.config.epochs)
        
        shape = (X_train.shape[1], X_train.shape[2])

        self.buildModel(shape=shape)

        print(self.model.summary())

        self.compileModel(scaler_target, lr_scheduler)

        self.trainModel(X_train, y_train, validation_data=(X_test, y_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="add specfic values for training")
    parser.add_argument('--name', type=str, default = "baseline", help='WandB Name for the run')
    parser.add_argument('--distance', type=int, default=750, help='Enter the distance up to which all data points should be included')
    parser.add_argument('--lstm_neurons', default=64, type=int, help='Define the number of neurons used in the lstm layers')
    parser.add_argument('--dense_neurons', default=16, type=int, help='Define the number of neurons used in the dense layers')
    parser.add_argument('--learning_rate', default=0.001 ,type=float, help='Define the learning rate of the training')
    parser.add_argument('--decay_rate', default=0.96, type=float, help='Define the decay rate of the learning rate')
    parser.add_argument('--activation_hidden', default='relu', type=str, help='Activastion function for hidden layers')
    parser.add_argument('--activation_final', default='tanh', type=str, help='Activation function for final/output layer')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of batches')
    args = parser.parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="pos-ba",
        name=args.name,

        # track hyperparameters and run metadata with wandb.config
        config={
            "lstm_neurons": args.lstm_neurons,
            "dense_neurons": args.dense_neurons,
            "distance": args.distance,
            "learning_rate": args.learning_rate,
            "learning_rate_decay": args.decay_rate,
            "epochs": args.epochs,
            "optimizer": "Adam",
            "loss": "mean squarred error",
            "batch_size": args.batch_size,
            "activation_hidden": args.activation_hidden,
            "activation_final": args.activation_final,
        }
    )

    model = RegressionModel()

    model.main()

    wandb.finish()