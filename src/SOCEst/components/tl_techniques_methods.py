import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Bidirectional, GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense, LSTM, GRU

# from numpy.lib.financial import rate
# from keras import optimizers
# from keras_tuner import RandomSearch, BayesianOptimization

tf.random.set_seed(314)
np.random.seed(314)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Define MMD loss function
def maximum_mean_discrepancy(y_true, y_pred):
    # Compute MMD between true labels and predicted labels
    kernel = tf.exp(-0.5 * tf.square(tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_true, axis=0)))
    kernel_mean = tf.reduce_mean(kernel, axis=[0, 1])
    kernel = tf.exp(-0.5 * tf.square(tf.expand_dims(y_pred, axis=1) - tf.expand_dims(y_pred, axis=0)))
    kernel_mean += tf.reduce_mean(kernel, axis=[0, 1])
    kernel = tf.exp(-0.5 * tf.square(tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_pred, axis=0)))
    kernel_mean -= 2 * tf.reduce_mean(kernel, axis=[0, 1])
    return tf.reduce_sum(kernel_mean)

def coral_loss(y_true, y_pred):
    source_covariance = tf.linalg.matmul(tf.transpose(y_true), y_true)
    target_covariance = tf.linalg.matmul(tf.transpose(y_pred), y_pred)

    coral_loss = tf.reduce_sum(tf.square(source_covariance - target_covariance))
    return coral_loss

    # source_covariance = tf.matmul(tf.transpose(target_feature), target_feature)
    # target_covariance = tf.matmul(tf.transpose(source_feature), source_feature)
    # loss = tf.reduce_sum(tf.square(source_covariance - target_covariance))
    # loss /= (4 * tf.cast(tf.shape(target_feature)[0], tf.float32) ** 2)
    # return loss

def transfer_learning1(X, y, model, model_opt):
    #Select the iptimizer 
    opt = tf.keras.optimizers.Adam(lr=0.001)

    #Select the loss 
    loss_f = tf.keras.losses.Huber()
    
    #Select the early stopping 
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])
    
    # Fine-tune All layers
    model.compile(optimizer=opt, loss=loss_f, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fine tune the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es], validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning2(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Freeze the Dense layers and fine-tune the RNN layers
    
    # Iterate over the layers in the model
    for i, layer in enumerate(model.layers):
        # Freeze the first two LSTM layers
        if isinstance(layer, Dense):
            layer.trainable = False
            print(f"Layer {i + 1} (RNN) frozen")

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning3(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Iterate over the layers in the model
    # Freeze all dense layers except for the last one
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = False

    # Unfreeze the last dense layer
    model.layers[-1].trainable = True

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning4(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Iterate over the layers in the model
    # Freeze all dense layers except for the last one
    for layer in model.layers:
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = False

    for layer in reversed(model.layers):
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = True
            break

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning5(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Iterate over the layers in the model
    # Freeze all dense layers except for the last one
    for layer in model.layers:
        layer.trainable = False

    for layer in reversed(model.layers):
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = True
            break
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = True
            break

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning6(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    for layer in model.layers:
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = False

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning7(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Iterate over the layers in the model
    # Freeze all dense layers except for the last one
    for layer in model.layers:
        layer.trainable = False

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = True
            break

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning8(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    for layer in model.layers:
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = False
            
    for layer in model.layers:
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = True
            break
    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning9(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers:
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = True
            break
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.trainable = True
            break
    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning10(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Fine-tune LSTM layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = False

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)

    # Apply MMD to the dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True
        else:
            layer.trainable = False

    # Compile the model with MMD loss function
    model.compile(optimizer=opt, loss=maximum_mean_discrepancy,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Train the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning11(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    for layer in model.layers:
        layer.trainable = False

    for layer in reversed(model.layers):
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = True
            break

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)

    # Apply MMD to the dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True
        else:
            layer.trainable = False

    # Compile the model with MMD loss function
    model.compile(optimizer=opt, loss=maximum_mean_discrepancy,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Train the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning12(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Apply MMD to the dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True
        else:
            layer.trainable = False

    # Compile the model with MMD loss function
    model.compile(optimizer=opt, loss=maximum_mean_discrepancy,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Train the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model


def transfer_learning13(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers:
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = True
            break

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)

    # Apply MMD to the dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True
        else:
            layer.trainable = False

    # Compile the model with MMD loss function
    model.compile(optimizer=opt, loss=maximum_mean_discrepancy,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Train the model
    history = model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
                        validation_split=model_opt['validation_split'], verbose=0)

    return model

def transfer_learning14(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Fine-tune LSTM layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = False

    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)

    # Apply CORAL to the dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer='adam', loss=coral_loss, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)  # Adjust the number of epochs as needed

    return model

def transfer_learning15(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    for layer in model.layers:
        layer.trainable = False

    for layer in reversed(model.layers):
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = True
            break

    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)

    # Apply CORAL to the dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer='adam', loss=coral_loss, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)  # Adjust the number of epochs as needed

    return model

def transfer_learning16(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    # Apply CORAL to the dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer='adam', loss=coral_loss, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)  # Adjust the number of epochs as needed

    return model

def transfer_learning17(X, y, model, model_opt):
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss_f = tf.keras.losses.Huber()
    es = EarlyStopping(monitor='val_loss', patience=model_opt['patience'])

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers:
        if isinstance(layer, LSTM) or isinstance(layer, GRU) or isinstance(layer, Bidirectional):
            layer.trainable = True
            break

    # Compile the model with the initial loss function
    model.compile(optimizer=opt, loss=loss_f,
                  metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    # Fit the model
    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)

    # Apply CORAL to the dense layers
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.trainable = True
        else:
            layer.trainable = False


    model.compile(optimizer='adam', loss=coral_loss, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    model.fit(X, y, epochs=model_opt['epochs'], batch_size=model_opt['batch_size'], callbacks=[es],
              validation_split=model_opt['validation_split'], verbose=0)  # Adjust the number of epochs as needed

    return model