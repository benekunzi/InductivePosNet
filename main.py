import pandas as pd
import numpy as np
import os 
import glob
from typing import List, Tuple
import pickle
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, BatchNormalization, Add, Dropout
try:
    from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay, CosineDecay
except ImportError:
    from keras.optimizers.schedules import ExponentialDecay, CosineDecay
from keras.callbacks import CSVLogger
from keras.initializers import HeUniform
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
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
    
    def __process_csv_files_and_extend(self, csv_block: List[str], n= 3) -> Tuple[pd.DataFrame, int]:
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
        size_df = None

        curr_dz = float(csv_block[0].split("_dz")[1].split("/")[0]) # extract current z-height of dataset
        for idx1, sensor_file in enumerate(csv_block):
            data_list = [] # create a list to store the data

            curr_sensor = sensor_file.split("_")[-1].split(".")[0]  # extract current sensor's name in dataset

            df = pd.read_csv(sensor_file,skiprows=28,sep=",",index_col=0)
            df.index = pd.to_numeric(df.index)
            df.columns = pd.to_numeric(df.columns)

            # Original array
            original_array = df.to_numpy(dtype='float')
            index_array = df.index.to_numpy(dtype='float')
            header_array = df.columns.to_numpy(dtype='float')

            # Extend the array
            extended_array = self.__extend_array(original_array, n)
            extended_array = extended_array.T
            extended_array = self.__extend_array(extended_array, n)
            extended_array = extended_array.T
            
            extended_index_array = self.__extend_array(index_array, n)
            extended_header_array = self.__extend_array(header_array, n)

            # merge the extended arrays to one dataframe
            df_extended = pd.DataFrame(extended_array, 
                                    index = extended_index_array, 
                                    columns = extended_header_array)
            
            df_extended.index.name = 'rows\\cols'

            if size_df is None:
                size_df = df_extended.shape

            for row_label, row in df_extended.iterrows():
                for col_label, value in row.items():
                    data_list.append((value, col_label, row_label, curr_dz, curr_sensor))

            tmp_dfs.append(pd.DataFrame(data_list, columns=["data","x","y","z","sensor"]))

            for idx2, tmp_df in enumerate(tmp_dfs):
                sensor_name = tmp_df["sensor"][0]
                merged_df[sensor_name] = tmp_df["data"]
            merged_df["x"] = tmp_df["x"]
            merged_df["y"] = tmp_df["y"]
            merged_df["z"] = tmp_df["z"]
            
        return (merged_df, size_df)

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

        if self.config.extended:
            print("extends dataframe for training")
            for _, z_height_group in enumerate(csv_files): # loop through each group of files (A4T, A5T, A6T, A7T)
                merged_df_list.append(self.__process_csv_files_and_extend(z_height_group)[0])
        else:
            for _, z_height_group in enumerate(csv_files): # loop through each group of files (A4T, A5T, A6T, A7T)
                merged_df_list.append(self.__process_csv_files(z_height_group))

        final_combined_df = pd.concat(merged_df_list, ignore_index=True)

        return final_combined_df

    def load_scaler(self, path_scaler: str) -> MinMaxScaler | StandardScaler:
        with open(path_scaler, 'rb') as file:
            scaler = pickle.load(file)

        return scaler

    def cut_dataframe(self, dataframe: pd.DataFrame, distance: int) -> pd.DataFrame:
        # Calculate Euclidean distance
        dataframe['distance'] = np.sqrt(dataframe['x']**2 + dataframe['y']**2)

        # Filter the DataFrame based on the distance threshold
        filtered_df = dataframe[dataframe['distance'] <= distance]

        # Drop the 'distance' column if you don't need it anymore and return
        return filtered_df.drop(columns=['distance'])

    def generate_inputData(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generates input data"""
        return dataframe[["A4T", "A5T", "A6T", "A7T"]]
    
    def generate_targetData(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generates target data"""
        return dataframe[["x", "y"]]

    def scale(self, scaler: MinMaxScaler | StandardScaler, data: pd.DataFrame) -> np.ndarray:
        return scaler.transform(data)

    def generate_weights(self, scaled_target_data: np.ndarray) -> np.ndarray:
        diff_from_1 = np.linalg.norm([-0.5, 0.0] - scaled_target_data, axis=1)
        output_weights = np.exp(-(diff_from_1/0.5)**2)

        return output_weights

    def createScheduler(self, size: int, inital_learning_rate: float) -> CosineDecay:
        print("decay steps:", np.ceil((size * (1 - self.validation_split)) / self.config.batch_size) * self.config.epochs)
        print(size, self.config.batch_size, self.config.epochs)
        return CosineDecay(
            initial_learning_rate = inital_learning_rate,
            decay_steps = np.ceil((size * (1 - self.validation_split)) / self.config.batch_size) * self.config.epochs,
        )
        # return ExponentialDecay(
        #     initial_learning_rate= self.config.learning_rate,
        #     decay_steps= (size // self.config.batch_size),
        #     decay_rate= self.config.learning_rate_decay
        # )

    def buildModel(self):
        if not self.config.dropout:
            print("Build model of type MLP")
        elif self.config.dropout:
            print("Build model of type MLP with dropout")
        n_features = 4
        self.model.add(Input(shape= (n_features), name='Dense_Input'))
        for i in range(0, self.config.layers):
            self.model.add(Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name=f'Dense_{i}'))
            self.model.add(Activation(self.config.activation_hidden, name=f'relu_{i}'))
            self.model.add(Dropout(rate=self.config.dropout_value if self.config.dropout else 0, name=f'dropout_{i}'))
            self.model.add(BatchNormalization())
        self.model.add(Dense(2, name='Dense_Out'))
        self.model.add(Activation(self.config.activation_final, name='activation_Out'))

    def buildModelWithRegularization(self):
        if not self.config.dropout:
            print("Build model of type MLP with Regularization")
        elif self.config.dropout:
            print("Build model of type MLP with Regularization and dropout")
        
        n_features = 4
        l2 = self.config.regularization_value
        self.model.add(Input(shape= (n_features), name='Dense_Input'))
        for i in range(0, self.config.layers):
            self.model.add(Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name=f'Dense_{i}', kernel_regularizer=L2(l2=l2)))
            self.model.add(Activation(self.config.activation_hidden, name=f'relu_{i}'))
            self.model.add(Dropout(rate=self.config.dropout_value if self.config.dropout else 0, name=f'dropout_{i}'))
            self.model.add(BatchNormalization())
        self.model.add(Dense(2, name='Dense_Out', kernel_regularizer=L2(l2=l2)))
        self.model.add(Activation(self.config.activation_final, name='activation_Out'))

    def buildModelResnet(self):
        print("Build model of type ResNet")
        n_features = 4
        inputs = Input(shape= (n_features), name='Dense_Input')
        x = inputs
        x = Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name=f'Dense_0')(x)
        x = Activation(self.config.activation_hidden, name=f'relu_0')(x)
        x = BatchNormalization()(x)
        
        for i in range(1, self.config.layers, 2):
            x_temp = x
            x = Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name=f'Dense_{i}')(x)
            x = Activation(self.config.activation_hidden, name=f'relu_{i}')(x)
            x = BatchNormalization()(x)
            x = Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name=f'Dense_{i+1}')(x)
            x = Activation(self.config.activation_hidden, name=f'relu_{i+1}')(x)
            x = BatchNormalization()(x)
            x = Add()([x, x_temp])

        x = Dense(2, name='Dense_Out')(x)
        outputs = Activation(self.config.activation_final, name='Sigmoid_Out')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name="ChatGPT-85")

    def buildModelResnetWithRegularization(self):
        print("Build model of type ResNet with Regularization")
        n_features = 4
        inputs = Input(shape= (n_features), name='Dense_Input')
        x = inputs
        x = Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name=f'Dense_0', kernel_regularizer='l2')(x)
        x = Activation(self.config.activation_hidden, name=f'Relu_0')(x)
        x = Dropout(rate=self.config.dropout_value if self.config.dropout else 0, name="Dropout_0")(x)
        x = BatchNormalization()(x)
        
        for i in range(1, self.config.layers, 2):
            x_temp = x
            x = Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name=f'Dense_{i}', kernel_regularizer='l2')(x)
            x = Activation(self.config.activation_hidden, name=f'Relu_{i}')(x)
            x = Dropout(rate=self.config.dropout_value if self.config.dropout else 0, name=f'Dropout_{i}')(x)
            x = BatchNormalization()(x)
            x = Dense(self.config.n_neurons, kernel_initializer= HeUniform(), name=f'Dense_{i+1}', kernel_regularizer='l2')(x)
            x = Activation(self.config.activation_hidden, name=f'Relu_{i+1}')(x)
            x = Dropout(rate=self.config.dropout_value if self.config.dropout else 0, name=f'Dropout_{i+1}')(x)
            x = BatchNormalization()(x)
            x = Add()([x, x_temp])

        x = Dense(2, name='Dense_Out', kernel_regularizer='l2')(x)
        outputs = Activation(self.config.activation_final, name='Sigmoid_Out')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name="ChatGPT-85")

    def compileModel(self, prescaler_target_data: MinMaxScaler, lr_scheduler: CosineDecay):
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
                # metrics.MaxXYNormRealWorld(scaler=prescaler_target_data)
            ],
            weighted_metrics= []
        )

    def trainModel(self, scaled_input_data: np.ndarray, scaled_target_data: np.ndarray, outputWeights: np.ndarray, validation_data) -> None:
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
        callbacks = [
            csv_logger,
            WandbMetricsLogger(log_freq=1),
            # mcp_save
        ]

        history =self.model.fit(
            x = scaled_input_data,
            y = scaled_target_data, 
            epochs = self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks = callbacks,
            sample_weight= pd.Series(outputWeights).to_frame('weights'),
            shuffle=True,
            validation_data=validation_data
        )

        # with open(f'history/{identifier}.pkl', 'wb') as file:
        #     pickle.dump(history, file)


    def train_test_split(self, features: pd.DataFrame, target: pd.DataFrame):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.validation_split, random_state=42)

        return (X_train, X_test, y_train, y_test)

    def main(self):
        dataframe_total = self.generate_dataframe()

        X_train, X_test, y_train, y_test = self.train_test_split(dataframe_total[['A4T', 'A5T', 'A6T', 'A7T']], dataframe_total[['x', 'y']])

        assert isinstance(X_train, pd.DataFrame), "dings bums no dataframe"

        scaler_input = self.load_scaler("prescaler_input_data_Range-1-1.pkl")
        scaler_target = self.load_scaler("prescaler_target_data_Range-1-1.pkl")

        # first training iteration
        dataframe = self.cut_dataframe(dataframe_total, self.center_distance)
        df_size = len(dataframe.index)

        train_combined = pd.concat([X_train, y_train], axis=1)
        test_combined = pd.concat([X_test, y_test], axis=1)

        train_combined_center = self.cut_dataframe(train_combined, self.center_distance)
        test_combined_center = self.cut_dataframe(test_combined, self.center_distance)

        X_train_center = train_combined_center[['A4T', 'A5T', 'A6T', 'A7T']]
        X_test_center = test_combined_center[['A4T', 'A5T', 'A6T', 'A7T']]
        y_train_center = train_combined_center[['x', 'y']]
        y_test_center = test_combined_center[['x', 'y']]

        # input_data = self.generate_inputData(dataframe)
        # target_data = self.generate_targetData(dataframe)

        X_train_center = self.scale(scaler_input, X_train_center)
        y_train_center = self.scale(scaler_target, y_train_center)

        X_test_center = self.scale(scaler_input, X_test_center)
        y_test_center = self.scale(scaler_target, y_test_center)

        weights = self.generate_weights(y_train_center)
        lr_scheduler = self.createScheduler(df_size, inital_learning_rate=self.config.learning_rate)

        if self.config.resnet and self.config.regularization:
            self.buildModelResnetWithRegularization()
        elif self.config.resnet and not self.config.regularization:
            self.buildModelResnet()
        elif not self.config.resnet and self.config.regularization:
            self.buildModelWithRegularization()
        elif not self.config.resnet and not self.config.regularization:
            self.buildModel()

        print(self.model.summary())

        self.compileModel(scaler_target, lr_scheduler)

        self.trainModel(X_train_center, y_train_center, weights, validation_data=(X_test_center, y_test_center))

        # second training iteration

        dataframe = self.cut_dataframe(dataframe_total, self.config.distance)
        df_size = len(dataframe.index)

        train_combined = pd.concat([X_train, y_train], axis=1)
        test_combined = pd.concat([X_test, y_test], axis=1)

        train_combined_center = self.cut_dataframe(train_combined, self.config.distance)
        test_combined_center = self.cut_dataframe(test_combined, self.config.distance)

        X_train_center = train_combined_center[['A4T', 'A5T', 'A6T', 'A7T']]
        X_test_center = test_combined_center[['A4T', 'A5T', 'A6T', 'A7T']]
        y_train_center = train_combined_center[['x', 'y']]
        y_test_center = test_combined_center[['x', 'y']]

        X_train_center = self.scale(scaler_input, X_train_center)
        y_train_center = self.scale(scaler_target, y_train_center)

        X_test_center = self.scale(scaler_input, X_test_center)
        y_test_center = self.scale(scaler_target, y_test_center)

        # input_data = self.generate_inputData(dataframe)
        # target_data = self.generate_targetData(dataframe)

        weights = self.generate_weights(y_train_center)

        smaller_learning_rate = 0.0025
        lr_scheduler = self.createScheduler(df_size, inital_learning_rate=smaller_learning_rate)

        self.compileModel(scaler_target, lr_scheduler)
        self.trainModel(X_train_center, y_train_center, weights, validation_data=(X_test_center, y_test_center))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="add specfic values for training")
    parser.add_argument('--name', type=str, default = "baseline", help='WandB Name for the run')
    parser.add_argument('--distance', type=int, default=250, help='Enter the distance up to which all data points should be included')
    parser.add_argument('--n_neurons', default=300, type=int, help='Define the number of neurons used in the hidden layers')
    parser.add_argument('--n_layers', default=2, type=int, help='Define the number of layers')
    parser.add_argument('--learning_rate', default=0.001 ,type=float, help='Define the learning rate of the training')
    parser.add_argument('--decay_rate', default=0.96, type=float, help='Define the decay rate of the learning rate')
    parser.add_argument('--activation_hidden', default='relu', type=str, help='Activastion function for hidden layers')
    parser.add_argument('--activation_final', default='sigmoid', type=str, help='Activation function for final/output layer')
    parser.add_argument('--epochs', default=180, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of batches')
    parser.add_argument('--resnet', default=False, type=bool, help="Using resnet if true and if false using mlp")
    parser.add_argument('--regularization', default=False, type=bool, help="Applies l2 regularization to the net on all layers except input layer")
    parser.add_argument('--regularization_value', default=0.01, type=float, help="Applies the given value to L2")
    parser.add_argument('--dropout', default=False, type=bool, help="Applying dropout to tge layers")
    parser.add_argument('--dropout_value', default=0.2, type=float, help="Define the amount of dropout applied to all layers")
    parser.add_argument('--extended', default=False, type=bool, help="Extends the whole dataset used for training and validation")
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
            "activation_final": args.activation_final,
            "layers": args.n_layers,
            "resnet": args.resnet,
            "regularization": args.regularization,
            "regularization_value": args.regularization_value,
            "dropout": args.dropout, 
            "dropout_value": args.dropout_value,
            "extended": args.extended
        }
    )

    model = RegressionModel()

    model.main()

    wandb.finish()