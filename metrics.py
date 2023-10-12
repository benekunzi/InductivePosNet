import tensorflow as tf

class MeanXYSquarredErrorRealWorld(tf.keras.metrics.MeanSquaredError):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="mseXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        return (y - self.scaler.min_) / self.scaler.scale_

    def update_state(self, y_pred, y_target, *args, **kwargs):
        super().update_state(self.inverse_transform(y_pred), 
                             self.inverse_transform(y_target), *args, **kwargs)
    
class MeanXYAbsoluteErrorRealWorld(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="maeXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        return (y - self.scaler.min_) / self.scaler.scale_

    def update_state(self, y_pred, y_target, *args, **kwargs):
        super().update_state(self.inverse_transform(y_pred), 
                             self.inverse_transform(y_target), *args, **kwargs)
        
class MeanXYRootSquarredErrorRealWorld(tf.keras.metrics.RootMeanSquaredError):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="rmseXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        return (y - self.scaler.min_) / self.scaler.scale_

    def update_state(self, y_pred, y_target, *args, **kwargs):
        super().update_state(self.inverse_transform(y_pred), 
                             self.inverse_transform(y_target), *args, **kwargs)
        
class MeanXYNormRealWorld(tf.keras.metrics.Mean):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="normXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        return (y - self.scaler.min_) / self.scaler.scale_

    def update_state(self, y_pred, y_target, *args, **kwargs):

        y_pred = self.inverse_transform(y_pred)
        y_target = self.inverse_transform(y_target)
        error = y_target - y_pred
        norm = tf.norm(error, axis=1)

        super().update_state(norm, *args, **kwargs)

class CenterMeanXYNormRealWorld(tf.keras.metrics.Metric):
    def __init__(self, scaler, *args, **kwargs):
        super().__init__(name="centernormXYReal", *args, **kwargs)
        self.scaler = scaler

    def inverse_transform(self, y):
        # original sklearn
        return (y - self.scaler.min_) / self.scaler.scale_

    def update_state(self, y_pred, y_target, *args, **kwargs):

        y_pred = self.inverse_transform(y_pred)
        y_target = self.inverse_transform(y_target)
        error = y_target - y_pred
        norm = tf.norm(error, axis=1)

        distance = tf.norm(y_target, axis=1)
        selection = tf.boolean_mask(norm, distance < 50)

        super().update_state(selection, *args, **kwargs)