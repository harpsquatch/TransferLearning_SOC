import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Bidirectional, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner import RandomSearch, BayesianOptimization, Hyperband


class TransferLearningModel:
    def __init__(self):
        tf.random.set_seed(314)
        np.random.seed(314)

    # Define MMD loss function
    def maximum_mean_discrepancy(self,y_true, y_pred):
        # Compute MMD between true labels and predicted labels
        kernel = tf.exp(-0.5 * tf.square(tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_true, axis=0)))
        kernel_mean = tf.reduce_mean(kernel, axis=[0,1])
        kernel = tf.exp(-0.5 * tf.square(tf.expand_dims(y_pred, axis=1) - tf.expand_dims(y_pred, axis=0)))
        kernel_mean += tf.reduce_mean(kernel, axis=[0,1])
        kernel = tf.exp(-0.5 * tf.square(tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_pred, axis=0)))
        kernel_mean -= 2 * tf.reduce_mean(kernel, axis=[0,1])
        return tf.reduce_sum(kernel_mean)

    def transfer_learning3(self, X, y, model,optimizer,loss_f,es ,epochs,batch_size,validation_split):
        # Define new model with LSTM layers and output Dense layer
        # Fine-tune LSTM layers
        for layer in model.layers:
            if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
                layer.trainable = True
                print("LSTM layer found")
            else:
                layer.trainable = False
                print("Dense layer found")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        # Apply MMD to the dense layers
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.trainable = True
            else:
                layer.trainable = False

        # Compile the model with MMD loss function
        model.compile(optimizer=optimizer, loss=self.maximum_mean_discrepancy, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        # Train the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning1(self, X, y, model,optimizer,loss_f,es ,epochs,batch_size,validation_split):
        # Define new model with LSTM layers and output Dense layer
        # Fine-tune All layers
        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fine tune the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning2(self, X, y, model,optimizer,loss_f,es ,epochs,batch_size,validation_split):
        # Define new model with LSTM layers and output Dense layer
        # Freeze the first two LSTM and Dense layers and fine-tune the rest layers
        # Iterate over the layers in the model
        if len(model.layers) == 5:
            nl = 1
        elif len(model.layers) == 7:
            nl = 2

        for i, layer in enumerate(model.layers):
            # Freeze the first two LSTM layers
            if i < nl and (isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional)):
                layer.trainable = False
                print(f"Layer {i+1} (LSTM) frozen")
            # Freeze the Dropout layer
            elif isinstance(layer, Dropout):
                layer.trainable = False
                print("Dropout layer frozen")
            # Freeze the first two Dense layers
            elif i < len(model.layers)-1 and isinstance(layer, Dense):
                layer.trainable = False
                print(f"Layer {i+1} (Dense) frozen")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning4(self, X, y, model,optimizer,loss_f,es ,epochs,batch_size,validation_split):

        # Define new model with LSTM layers and output Dense layer
        # Freeze all LSTM layers and fine tune Dense layers
        for layer in model.layers:
            if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
                layer.trainable = False
                print("LSTM layer found")
            else:
                layer.trainable = True
                print("Dense layer found")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning5(self, X, y, model,optimizer,loss_f,es ,epochs,batch_size,validation_split):

        # Define new model with LSTM layers and output Dense layer
        # Freeze all layers and fine-tune the third recurrent layer
        # Iterate over the layers in the model
        if len(model.layers) == 5:
            nl = 1 # the 2nd 
        elif len(model.layers) == 7:
            nl = 2 # the 3rd
        for i, layer in enumerate(model.layers):
            # Freeze the first two LSTM layers
            if i == nl and (isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional)):
                layer.trainable = True
                print(f"Layer {i+1} (LSTM) frozen")
            # Freeze the Dropout layer
            else:
                layer.trainable = False
                print(f"Layer {i+1} (Dense) frozen")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning6(self, X, y, model,optimizer,loss_f,es ,epochs,batch_size,validation_split):

        # Define new model with LSTM layers and output Dense layer
        # Freeze 2nd LSTM layer
        
        if len(model.layers) == 5:
            nl = 1 # the 2nd 
        elif len(model.layers) == 7:
            nl = 1 # the 2nd    
        for i, layer in enumerate(model.layers):
            if i == nl and isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
                layer.trainable = False
                print("LSTM layer found")
            else:
                layer.trainable = False
                print("Dense layer found")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        # Apply MMD to the dense layers
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.trainable = True
            else:
                layer.trainable = False

        # Compile the model with MMD loss function
        model.compile(optimizer=optimizer, loss=self.maximum_mean_discrepancy, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        # Train the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning7(self, X, y, model,optimizer,loss_f,es ,epochs,batch_size,validation_split):
        # Define new model with LSTM layers and output Dense layer
        # Freeze 1st LSTM layer
        
        if len(model.layers) == 5:
            nl = 1 # the 2nd 
        elif len(model.layers) == 7:
            nl = 0 # the 1st    
        for i, layer in enumerate(model.layers):
            if i == nl and isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
                layer.trainable = False
                print("LSTM layer found")
            else:
                layer.trainable = False
                print("Dense layer found")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        # Apply MMD to the dense layers
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.trainable = True
            else:
                layer.trainable = False

        # Compile the model with MMD loss function
        model.compile(optimizer=optimizer, loss=self.maximum_mean_discrepancy, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        # Train the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning8(self, X, y, model,optimizer,loss_f,es ,epochs,batch_size,validation_split):

        # Define new model with LSTM layers and output Dense layer
        # Freeze all layers and fine-tune the first & third recurrent layer
        # Iterate over the layers in the model
        if len(model.layers) == 5:
            nl = 1 # the 2nd 
        elif len(model.layers) == 7:
            nl = 1 # the 2nd

        
        for i, layer in enumerate(model.layers):
            # Freeze the first two LSTM layers
            if i == nl and (isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional)):
                layer.trainable = False
                print(f"Layer {i+1} (LSTM) frozen")
            # Freeze the Dropout layer
            else:
                layer.trainable = False
                print(f"Layer {i+1} (Dense) frozen")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning9(self, X, y, model,optimizer,loss_f,es,epochs,batch_size,validation_split):

        # Define new model with LSTM layers and output Dense layer
        # Freeze 3st LSTM layer
        
        if len(model.layers) == 5:
            nl = 1 # the 2nd 
        elif len(model.layers) == 7:
            nl = 2 # the 3rd
        for i, layer in enumerate(model.layers):
            if i == nl and isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
                layer.trainable = False
                print("LSTM layer found")
            else:
                layer.trainable = False
                print("Dense layer found")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        # Apply MMD to the dense layers
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.trainable = True
            else:
                layer.trainable = False

        # Compile the model with MMD loss function
        model.compile(optimizer=optimizer, loss=self.maximum_mean_discrepancy, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

        # Train the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model

    def transfer_learning10(self, X, y, model,optimizer,loss_f,es,epochs,batch_size,validation_split):

        # Define new model with LSTM layers and output Dense layer
        # Freeze all layers 
        # Iterate over the layers in the model
        
        for i, layer in enumerate(model.layers):
            # Freeze the first two LSTM layers
            if (isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional)):
                layer.trainable = False
                print(f"Layer {i+1} (LSTM) frozen")
            # Freeze the Dropout layer
            else:
                layer.trainable = False
                print(f"Layer {i+1} (Dense) frozen")

        # Compile the model with the initial loss function
        model.compile(optimizer=optimizer, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        # Fit the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[es], validation_split=validation_split)

        return model