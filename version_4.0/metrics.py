import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

distance_center = 200

class MeanXYSquarredErrorRealWorld(tf.keras.metrics.MeanSquaredError):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="mseXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        if type(self.scaler) == MinMaxScaler: 
            return (y - self.scaler.min_) / self.scaler.scale_
        elif type(self.scaler) == StandardScaler: 
            y *= self.scaler.scale_
            # y *= self.scaler.mean_
            return y

    def update_state(self, y_pred, y_target, *args, **kwargs):
        super().update_state(self.inverse_transform(y_pred), 
                             self.inverse_transform(y_target), *args, **kwargs)
    
class MeanXYAbsoluteErrorRealWorld(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="maeXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        if type(self.scaler) == MinMaxScaler: 
            return (y - self.scaler.min_) / self.scaler.scale_
        elif type(self.scaler) == StandardScaler: 
            y *= self.scaler.scale_
            # y *= self.scaler.mean_
            return y

    def update_state(self, y_pred, y_target, *args, **kwargs):
        super().update_state(self.inverse_transform(y_pred), 
                             self.inverse_transform(y_target), *args, **kwargs)
        
class MeanXYRootSquarredErrorRealWorld(tf.keras.metrics.RootMeanSquaredError):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="rmseXYReal", *args, **kwargs)
        self.scaler = scaler
        
    def inverse_transform(self, y):
        # original sklearn
        if type(self.scaler) == MinMaxScaler: 
            return (y - self.scaler.min_) / self.scaler.scale_
        elif type(self.scaler) == StandardScaler: 
            y *= self.scaler.scale_
            # y *= self.scaler.mean_
            return y

    def update_state(self, y_pred, y_target, *args, **kwargs):
        super().update_state(self.inverse_transform(y_pred), 
                             self.inverse_transform(y_target), *args, **kwargs)
        
class MeanXYNormRealWorld(tf.keras.metrics.Mean):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="normXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        if type(self.scaler) == MinMaxScaler: 
            return (y - self.scaler.min_) / self.scaler.scale_
        elif type(self.scaler) == StandardScaler: 
            y *= self.scaler.scale_
            # y *= self.scaler.mean_
            return y

    def update_state(self, y_pred, y_target, *args, **kwargs):

        y_pred = self.inverse_transform(y_pred)
        y_target = self.inverse_transform(y_target)
        error = y_target - y_pred
        norm = tf.norm(error, axis=1)

        super().update_state(norm, *args, **kwargs)

class CenterMeanXYNormRealWorld(tf.keras.metrics.Mean):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="centernormXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        if type(self.scaler) == MinMaxScaler: 
            return (y - self.scaler.min_) / self.scaler.scale_
        elif type(self.scaler) == StandardScaler: 
            y *= self.scaler.scale_
            # y *= self.scaler.mean_
            return y

    def update_state(self, y_pred, y_target, *args, **kwargs):

        y_pred = self.inverse_transform(y_pred)
        y_target = self.inverse_transform(y_target)
        error = y_target - y_pred
        norm = tf.norm(error, axis=1)

        distance = tf.norm(y_target, axis=1)
        selection = tf.boolean_mask(norm, distance < distance_center)

        super().update_state(selection, *args, **kwargs)


class CenterMaxXYNormRealWorld(tf.keras.metrics.Metric):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="CenterMaxNormXYReal", *args, **kwargs)
        self.scaler = scaler
        self.max = self.add_weight("max", initializer=tf.keras.initializers.Zeros(), dtype=tf.float32)

    def inverse_transform(self, y):
        if type(self.scaler) == MinMaxScaler: 
            return (y - self.scaler.min_) / self.scaler.scale_
        elif type(self.scaler) == StandardScaler: 
            y *= self.scaler.scale_
            # y *= self.scaler.mean_
            return y

    def update_state(self, y_pred, y_target, *args, **kwargs):

        y_pred = self.inverse_transform(y_pred)
        y_target = self.inverse_transform(y_target)
        error = y_target - y_pred
        norm = tf.norm(error, axis=1) 

        distance = tf.norm(y_target, axis=1)
        selection = tf.boolean_mask(norm, distance < distance_center)

        norm_max = tf.math.reduce_max(selection)

        if norm_max > self.max:
            self.max.assign(norm_max)

    def result(self):
        return self.max

    def reset_state(self):
        self.max.assign(0.0)

class CenterWithinTresholdXYNormRealWorld(tf.keras.metrics.Mean):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="CenterWithinTresholdNormXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        if type(self.scaler) == MinMaxScaler: 
            return (y - self.scaler.min_) / self.scaler.scale_
        elif type(self.scaler) == StandardScaler: 
            y *= self.scaler.scale_
            # y *= self.scaler.mean_
            return y

    def update_state(self, y_pred, y_target, *args, **kwargs):

        y_pred = self.inverse_transform(y_pred)
        y_target = self.inverse_transform(y_target)
        error = y_target - y_pred
        norm = tf.norm(error, axis=1)

        distance = tf.norm(y_target, axis=1)
        selection = tf.boolean_mask(norm, distance < distance_center)

        percentage = (tf.math.reduce_mean(tf.cast(selection < 25, tf.float32))) * 100
        
        super().update_state(percentage, *args, **kwargs)