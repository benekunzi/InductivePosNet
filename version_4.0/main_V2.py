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
from keras.layers import Masking, Activation, Dense, BatchNormalization, Add, Dropout, LSTM, Masking
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
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

class RegressionModel():
    def __init__(self) -> None:
        self.directory_path = '01_measurement_data/training_set.csv'
        self.model = Sequential()
        self.config = wandb.config
        self.validation_split = 0.3
        self.center_distance = 100
        self.y_max = 820
        self.y_min = 0
        self.x_max = 820
        self.x_min = -620 
        self.start_points = []
        self.end_points = []
        self.merged_df_list = []
        self.x_coordinates = []
        self.y_coordinates = []
        self.mask_value = 2.0
        self.df = pd.DataFrame()
        self.repetition_counter = 0

    def __read_csv(self, csv_file: str) -> pd.DataFrame:
        return pd.read_csv(csv_file)
    
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
    
    def __generate_start_points(self) -> None:
        for x in self.x_coordinates:
            if x <= self.x_max and x >= self.x_min:
                self.start_points.append([x, self.y_max])

    def __generate_end_points(self) -> None:
        for x in self.x_coordinates:
            if x <= self.x_max and x >= self.x_min:
                self.end_points.append([x, self.y_min])

    def __interpolate_array(self, arr, total_length):
        # Time points for existing data
        existing_time_points = np.linspace(0, 1, arr.shape[0])
        # Time points for interpolated data
        new_time_points = np.linspace(0, 1, total_length)
        # Interpolation function for x and y
        fx = interp1d(existing_time_points, arr[:,0], kind='linear')
        fy = interp1d(existing_time_points, arr[:,1], kind='linear')
        # Interpolated x and y
        xnew = fx(new_time_points)
        ynew = fy(new_time_points)
        # Return interpolated array
        return np.stack((xnew, ynew), axis=-1)
    
    def __quadratic_bezier(self, p0, p1, p2, number_of_steps=30):
        t = np.linspace(0, 1, number_of_steps)
        p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
        
        # Compute the first interpolation between p0 and p1, and p1 and p2
        p01 = (1 - t)[:, None] * p0 + t[:, None] * p1
        p12 = (1 - t)[:, None] * p1 + t[:, None] * p2
        
        # Compute the final interpolation between p01 and p12
        curve_points = (1 - t)[:, None] * p01 + t[:, None] * p12
        
        return curve_points
    
    def __generate_bezier_curves(self, number_of_steps= 30):
        bezier_curves = []
        for start in self.start_points:
            for end in self.end_points:
                control_point = [end[0]-20, int(start[1]/1.3)]
                bezier_curves.append(self.__quadratic_bezier(start, control_point, end, number_of_steps=number_of_steps))

        return np.array(bezier_curves)
    
    def __generate_bezier_piecewise(self, bezier_curves: np.ndarray, divisors: list) -> Tuple[list, int]:
        new_elements_all = []
        for divisor in divisors:
            new_elements = []
            for bezier in bezier_curves:
                for i in range(0, bezier_curves[0].shape[0], divisor):
                    new_elements.append(bezier[i:i+divisor, :])

            new_elements_all.append(np.array(new_elements))

        size = 0
        for d in new_elements_all:
            size += len(d)

        return (new_elements_all, size)
    
    def __extend_bezier_curves(self, bezier_curves, size, number_of_steps=30) -> np.ndarray: 
        extended_beziers = np.zeros((size, number_of_steps, 2))
        counter = 0

        for element_divisor in bezier_curves:
            for element in element_divisor:
                extended_beziers[counter] = self.__interpolate_array(element, number_of_steps)
                counter += 1

        return extended_beziers

    def __generate_bezier_curves_real(self, bezier_curves: np.ndarray) -> np.ndarray:
        x_grid, y_grid = np.meshgrid(self.x_coordinates, self.y_coordinates)
        coordinate_grid = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

        kd_tree = cKDTree(coordinate_grid)

        bezier_curves_real = []
        for bezier in bezier_curves:
            # Query KD-tree for nearest neighbor
            dists, indices = kd_tree.query(bezier, k=1)
            # Retrieve the actual coordinates from the grid
            nearest_coords = coordinate_grid[indices]
            bezier_curves_real.append(nearest_coords)

        return np.array(bezier_curves_real)

    def __generate_voltage_values_bezier(self, bezier_curves_real: np.ndarray, n_functions: int, number_of_steps=30) -> Tuple[np.ndarray, np.ndarray]:
        grouped = self.df.groupby(['x', 'y'])[['a4t', 'a5t', 'a6t', 'a7t']].apply(lambda x: x.values.tolist()).to_dict()

        for key in grouped:
            while len(grouped[key]) < self.repetition_counter:
                grouped[key].append([np.nan, np.nan, np.nan, np.nan])
    
        voltage_values = np.full((n_functions*self.repetition_counter, number_of_steps, 4), np.nan)
        counter = 0
        # Populate the array with the voltage values from the dictionary
        for k, bezier in enumerate(bezier_curves_real):
            for i, (x, y) in enumerate(bezier):
                # Check if the coordinate is in the dictionary
                if (x, y) in grouped:
                    for j, voltage_array in enumerate(grouped[(x, y)]):
                        voltage_values[counter+j, i, :] = voltage_array
            counter += self.repetition_counter

        coordinate_values = np.repeat(bezier_curves_real, repeats=self.repetition_counter, axis=0)
        coordinate_values = coordinate_values[:, -1, :]

        return (voltage_values, coordinate_values)
    
    def __generate_all_paths_bezier(self, scaled_input_data: np.ndarray, scaled_target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Assuming scaled_input_data and scaled_target_data are defined and available
        array_length = scaled_input_data.shape[0]
        curves_length = scaled_input_data.shape[1]
        input_feature_length = scaled_input_data.shape[2]  # Should be 4 based on your description

        # Pre-calculate the total number of entries after expansion
        total_entries = array_length * curves_length

        # Pre-allocate arrays with expected shapes
        expanded_input_shape = (total_entries, curves_length, input_feature_length)
        expanded_target_shape = (total_entries, scaled_target_data.shape[2])  # Assuming shape (62500, 50, 2) -> (total_entries, 2)

        # Initialize the pre-allocated arrays
        scaled_input_temp_full = np.zeros(expanded_input_shape)
        scaled_target_temp_full = np.zeros(expanded_target_shape)

        index = 0  # To keep track of the current position in the pre-allocated arrays
        for i in range(array_length):
            for j in range(curves_length):
                # Calculate the mask
                mask_length = curves_length - (j + 1)
                if mask_length > 0:
                    mask = np.full((mask_length, input_feature_length), 2)
                    part_with_data = scaled_input_data[i, :j+1]
                    # Place the mask at the beginning
                    scaled_input_temp_full[index, :mask_length] = mask
                    # Place the data after the mask
                    scaled_input_temp_full[index, mask_length:mask_length+j+1] = part_with_data
                else:
                    # If there's no need for a mask, fill the entire sequence with data
                    scaled_input_temp_full[index, -j-1:] = scaled_input_data[i, :j+1]

                scaled_target_temp_full[index] = scaled_target_data[i][j]
                index += 1

        return (scaled_input_temp_full, scaled_target_temp_full)

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

    def createScheduler(self, size: int, inital_learning_rate: float, epochs: int) -> CosineDecay:
        print("decay steps:", np.ceil((size * (1 - self.validation_split)) / self.config.batch_size) * epochs)
        print(size, self.config.batch_size, epochs)
        return CosineDecay(
            initial_learning_rate = inital_learning_rate,
            decay_steps = np.ceil((size * (1 - self.validation_split)) / self.config.batch_size) * epochs,
        )

    def buildModel(self, shape):
        self.model = Sequential()
        self.model.add(Masking(mask_value=self.mask_value, input_shape=shape, name="input_masked"))
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
                metrics.CenterWithinTresholdXYNormRealWorld(scaler=prescaler_target_data)
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

        self.df = self.__read_csv(self.directory_path)

        self.repetition_counter = self.df[(self.df['x'] == -620) & (self.df['y'] == -820)].shape[0]

        self.x_coordinates =  self.df["x"].unique()
        self.y_coordinates =  self.df["y"].unique()

        print(self.x_coordinates)
        print(self.y_coordinates)

        self.__generate_start_points()
        self.__generate_end_points()

        print(self.start_points[0])
        print(self.end_points[0])

        number_of_steps = 30
        divisors =[2, 3, 5,  6, 10, 15]
        bezier_curves = self.__generate_bezier_curves(number_of_steps=number_of_steps)
        print(len(bezier_curves))

        bezier_curves, size = self.__generate_bezier_piecewise(bezier_curves, divisors)
        bezier_curves = self.__extend_bezier_curves(bezier_curves, size, number_of_steps=number_of_steps)
        print(bezier_curves.shape)

        bezier_curves_real = self.__generate_bezier_curves_real(bezier_curves)
        print(bezier_curves_real.shape)

        n_functions = bezier_curves_real.shape[0]
        voltage_values, coordinate_values = self.__generate_voltage_values_bezier(bezier_curves_real, n_functions, number_of_steps=number_of_steps)

        # Expending array with standing values
        coordinate_standing = self.df[['x', 'y']].to_numpy()
        voltage_values_standing = self.df[['a4t', 'a5t', 'a6t', 'a7t']].to_numpy()
        voltage_values_standing = np.tile(voltage_values_standing[:, np.newaxis, :], (1, 30, 1))

        coordinate_values = np.concatenate([coordinate_values, coordinate_standing], axis=0)
        print(coordinate_values.shape)
        voltage_values = np.concatenate([voltage_values, voltage_values_standing], axis=0)
        print(voltage_values.shape)

        # print("shape of pure linear functions setup without extension")
        # print(voltage_values.shape, coordinate_values.shape)

        scaler_input = self.load_scaler("prescaler_input_data_Range-1-1.pkl")
        scaler_target = self.load_scaler("prescaler_target_data_Range-1-1.pkl")

        voltage_values_R = voltage_values.reshape(-1, voltage_values.shape[-1])
        scaled_input_data_R = scaler_input.transform(voltage_values_R)
        scaled_input_data = scaled_input_data_R.reshape(voltage_values.shape)

        scaled_target_data = scaler_target.transform(coordinate_values)

        # scaled_input_data, scaled_target_data = self.__generate_all_paths_bezier(scaled_input_data, scaled_target_data)

        print("shape of bezier curves")
        print(scaled_input_data.shape, scaled_target_data.shape)
        print("min and max values of scaled data")
        print(scaled_input_data.min(), scaled_input_data.max(), scaled_target_data.min(), scaled_target_data.max())

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