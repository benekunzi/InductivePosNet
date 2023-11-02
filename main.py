import pandas as pd
import numpy as np
import os 
import glob
from typing import List
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, BatchNormalization
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.callbacks import CSVLogger
from keras.initializers import HeUniform
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import metrics
import datetime
import time
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import argparse

class RegressionModel():
    def __init__(self) -> None:
        self.directory_path =  '/Users/benedictkunzmann/Desktop/Bachelorarbeit/Verlauf/Modelle/20230902_version_3.1/01_measurement_data'
        self.model = Sequential()
        self.config = wandb.config
    
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

    def __process_csv_files(self, csv_block: List[str]) -> pd.DataFrame:
        """ parameter csv_block needs to be a list of 4 filepath strings.
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
        for _, sensor_file in enumerate(csv_block):
            data_list = [] # create a list to store the data

            curr_sensor = sensor_file.split("_")[-1].split(".")[0]  # extract current sensor's name in dataset

            df = pd.read_csv(sensor_file,skiprows=28,sep=",",index_col=0)
            df.index = pd.to_numeric(df.index)
            df.columns = pd.to_numeric(df.columns)

            for row_label, row in df.iterrows():
                for col_label, value in row.items():
                    data_list.append((value, col_label, row_label, curr_dz, curr_sensor))

            tmp_dfs.append(pd.DataFrame(data_list, columns=["data","x","y","z","sensor"]))

            for _, tmp_df in enumerate(tmp_dfs):
                sensor_name = tmp_df["sensor"][0]
                merged_df[sensor_name] = tmp_df["data"]
            merged_df["x"] = tmp_df["x"]
            merged_df["y"] = tmp_df["y"]
            merged_df["z"] = tmp_df["z"]
            
        return merged_df
    
    def generate_dataframe(self) -> pd.DataFrame:
        """takes all csv files and combines them into a single dataframe. The order of the is [A4T, A5T, A6T, A7T, x, y, z]"""
        csv_files = []
        merged_df_list = []

        csv_files = self.__get_csv_paths_list(self.directory_path) # function call to get the filepathes of all measurement files

        for _, z_height_group in enumerate(csv_files): # loop through each group of files (A4T, A5T, A6T, A7T)
            merged_df_list.append(self.__process_csv_files(z_height_group))

        final_combined_df = pd.concat(merged_df_list, ignore_index=True)

        return final_combined_df

    def load_scaler(self, path_scaler: str) -> MinMaxScaler:
        with open(path_scaler, 'rb') as file:
            scaler = pickle.load(file)

        return scaler

    def cut_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # Calculate Euclidean distance
        dataframe['distance'] = np.sqrt(dataframe['x']**2 + dataframe['y']**2)

        # Filter the DataFrame based on the distance threshold
        filtered_df = dataframe[dataframe['distance'] <= self.config.distance]

        # Drop the 'distance' column if you don't need it anymore and return
        return filtered_df.drop(columns=['distance'])

    def generate_inputData(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generates input data"""
        return dataframe[["A4T", "A5T", "A6T", "A7T"]]
    
    def generate_targetData(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generates target data"""
        return dataframe[["x", "y"]]

    def scale(self, scaler: MinMaxScaler, data: pd.DataFrame) -> np.ndarray:
        return scaler.transform(data)

    def generate_weights(self, scaled_target_data: np.ndarray) -> np.ndarray:
        diff_from_1 = np.linalg.norm(0.5 - scaled_target_data, axis=1)
        output_weights = np.exp(-(diff_from_1/0.6)**2)

        return output_weights

    def createScheduler(self, size: int) -> ExponentialDecay:
        return ExponentialDecay(
            initial_learning_rate= self.config.learning_rate,
            decay_steps= (size // self.config.batch_size) * self.config.epochs,
            decay_rate= self.config.learning_rate_decay
        )

    def buildModel(self):
        n_features = 4
        self.model.add(Input(shape= (n_features), name='Dense_Input'))
        self.model.add(Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name='Dense_1'))
        self.model.add(Activation(self.config.activation_hidden, name='relu_1'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name='Dense_2'))
        self.model.add(Activation(self.config.activation_hidden, name='relu_2'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(2, name='Dense_Out'))
        self.model.add(Activation(self.config.activation_final, name='sigmoid_Out'))

    def compileModel(self, prescaler_target_data: MinMaxScaler, lr_scheduler: ExponentialDecay):
        optimizer = Adam(learning_rate= lr_scheduler)
        loss = MeanSquaredError()

        self.model.compile(optimizer=optimizer, 
                            loss=loss,
                            metrics=[
                                "mse",
                                metrics.CenterMeanXYNormRealWorld(scaler=prescaler_target_data),
                                metrics.MeanXYNormRealWorld(scaler=prescaler_target_data),
                                metrics.MeanXYSquarredErrorRealWorld(scaler=prescaler_target_data),  
                                metrics.MeanXYAbsoluteErrorRealWorld(scaler=prescaler_target_data),
                                metrics.MeanXYRootSquarredErrorRealWorld(scaler=prescaler_target_data)
                            ],
                            weighted_metrics= []
        )

    def trainModel(self, scaled_input_data: np.ndarray, scaled_target_data: np.ndarray, outputWeights: np.ndarray) -> None:
        timestamp = time.time()
        datetime_obj = datetime.datetime.fromtimestamp(timestamp)
        # Format the datetime object as a string in your desired format
        readable_time = datetime_obj.strftime('%Y-%m-%d%H:%M:%S')

        identifier = f'{wandb.run.id}' # wandb creates id for model and save it online

        mcp_save = ModelCheckpoint(f'trained_models/{identifier}.h5', 
                        save_best_only=True,
                        monitor='val_mse', 
                        mode='min')
        
        csv_logger = CSVLogger(f'logging/{identifier}.csv',
                                append=False,
                                separator=',')

        history = self.model.fit(
            x = scaled_input_data,
            y = scaled_target_data, 
            epochs = self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks = [
                csv_logger,
                WandbMetricsLogger(log_freq=1),
                # learning_callback, 
                # mcp_save
            ],
            sample_weight= pd.Series(outputWeights).to_frame('weights'),
            shuffle=True,
            validation_split= 0.3
        )

        with open(f'history/{identifier}.pkl', 'wb') as file:
            pickle.dump(history, file)

    def main(self):
        dataframe = self.generate_dataframe()

        scaler_input = self.load_scaler("prescaler_input_data.pkl")
        scaler_target = self.load_scaler("prescaler_target_data.pkl")

        dataframe = self.cut_dataframe(dataframe)
        df_size = dataframe.size

        input_data = self.generate_inputData(dataframe)
        target_data = self.generate_targetData(dataframe)

        scaled_input_data = self.scale(scaler_input, input_data)
        scaled_target_data = self.scale(scaler_target, target_data)

        weights = self.generate_weights(scaled_target_data)
        lr_scheduler = self.createScheduler(df_size)

        self.buildModel()

        self.compileModel(scaler_target, lr_scheduler)

        self.trainModel(scaled_input_data, scaled_target_data, weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="add specfic values for training")
    parser.add_argument('--name', type=str, default = "baseline", help='WandB Name for the run')
    parser.add_argument('--distance', type=int, default = 250, help='Enter the distance up to which all data points should be included')
    parser.add_argument('--n_neurons', default= 300, type=int, help='Define the number of neurons used in the hidden layers')
    parser.add_argument('--learning_rate', default=0.001 ,type=float, help='Define the learning rate of the training')
    parser.add_argument('--decay_rate', default=0.96, type=float, help='Define the decay rate of the learning rate')
    parser.add_argument('--activation_hidden', default='relu', type=str, help='Activastion function for hidden layers')
    parser.add_argument('--activation_final', default='sigmoid', type=str, help='Activation function for final/output layer')
    parser.add_argument('--epochs', default=180, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of batches')
    args = parser.parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="pos-ba",
        name=args.name,

        # track hyperparameters and run metadata with wandb.config
        config={
            "n_neurons": args.n_neurons,
            "distance": args.distance,
            "learning_rate": args.learning_rate,
            "learning_rate_decay": args.decay_rate,
            "epochs": args.epochs,
            "optimizer": "Adam",
            "loss": "mean squarred error",
            "batch_size": args.batch_size,
            "activation_hidden": args.activation_hidden,
            "activation_final": args.activation_final
        }
    )

    model = RegressionModel()

    model.main()

    wandb.finish()