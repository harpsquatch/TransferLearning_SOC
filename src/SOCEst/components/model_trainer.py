import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, GRU, TimeDistributed, RepeatVector, Layer, Input
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer
# from numpy.lib.financial import rate

# from keras import optimizers
from keras_tuner import RandomSearch, BayesianOptimization 

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, GRU, TimeDistributed, RepeatVector, Layer, Input, Bidirectional
from tensorflow.keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

from kerastuner import HyperModel
from SOCEst.entity.config_entity import (ModelTrainerConfig)   

from SOCEst.components.transfer_learning import TransferLearningModel


class modelHO_New(HyperModel):
     
    def __init__(self, config):
        
        self.input_dim = config.input_dim
        self.steps = config.steps
        self.num_features = config.num_features
        self.dense_out = config.dense_out
        
        self.numberOfLayers = config.numberOfLayers
        self.stepUnit = config.stepUnit
        self.maxUnits = config.maxUnits

        self.numberOfDenseLayers = config.numberOfDenseLayers
        self.stepDenseUnit = config.stepDenseUnit
        self.maxDenseUnits = config.maxDenseUnits
        
        self.maxDropout = config.maxDropout
        self.dropoutRateStep = config.dropoutRateStep
        self.layer = config.layer


    def build(self, hp):
        
        loss_f = tf.keras.losses.Huber()

        #Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
        
        # Select optimizer  
        opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)  

        # Select activation function    
        activationDense=hp.Choice('activationDense', values=['leaky_relu', 'gelu', 'swish', 'selu', 'linear'], default='leaky_relu')

        with tf.device("/gpu:0"):
            model = Sequential()
            model.add(tf.keras.Input(shape=self.input_dim))
            if self.layer == 'lstm':
                for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                    model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=self.stepUnit, max_value=self.maxUnits, step=self.stepUnit),return_sequences=True))
                model.add(LSTM(hp.Int('layer_2_neurons', min_value=self.stepUnit, max_value=self.maxUnits, step=self.stepUnit),return_sequences=False))
            elif self.layer == 'bilstm':
                for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                    model.add(Bidirectional(LSTM(hp.Int(f'lstm_{i}_units', min_value=self.stepUnit, max_value=self.maxUnits, step=self.stepUnit),return_sequences=True)))
                model.add(Bidirectional(LSTM(hp.Int('layer_2_neurons', min_value=self.stepUnit, max_value=self.maxUnits, step=self.stepUnit),return_sequences=False)))
            elif self.layer == 'gru':
                for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                    model.add(GRU(hp.Int(f'lstm_{i}_units', min_value=self.stepUnit, max_value=self.maxUnits, step=self.stepUnit),return_sequences=True))
                model.add(GRU(hp.Int('layer_2_neurons', min_value=self.stepUnit, max_value=self.maxUnits, step=self.stepUnit),return_sequences=False))
            elif self.layer == 'bigru':
                for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                    model.add(Bidirectional(GRU(hp.Int(f'lstm_{i}_units', min_value=self.stepUnit, max_value=self.maxUnits, step=self.stepUnit),return_sequences=True)))
                model.add(Bidirectional(GRU(hp.Int('layer_2_neurons', min_value=self.stepUnit, max_value=self.maxUnits, step=self.stepUnit),return_sequences=False)))
            #model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=self.maxDropout, step=self.dropoutRateStep)))
            for i in range(hp.Int('n_layersDense', 1, self.numberOfDenseLayers)):
                model.add(Dense(hp.Int(f'dense_{i}_units', min_value=self.stepDenseUnit, max_value=self.maxDenseUnits, step=self.stepDenseUnit), activation=activationDense))
            model.add(Dense(self.dense_out, activation=activationDense))
            model.compile(loss=loss_f, optimizer=opt, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
            #model.summary()
            return model

class modelHO_SEGAN_LSTM(HyperModel):

    def __init__(self, config):
        # super()._init_()
        
        self.input_dim = config.input_dim
        self.dense_out = config.dense_out

        self.numberOfLayers = config.numberOfLayers
        self.stepUnit = config.stepUnit
        self.maxUnits = config.maxUnits

        self.numberOfDenseLayers = config.numberOfDenseLayers
        self.stepDenseUnit = config.stepDenseUnit
        self.maxDenseUnits = config.maxDenseUnits

        self.maxDropout = config.maxDropout
        self.dropoutRateStep = config.dropoutRateStep

        self.layer = config.layer


    def build(self, hp):
        loss_f = tf.keras.losses.Huber()
        opt = tf.keras.optimizers.Adam(lr=0.00001)
        with tf.device("/gpu:0"):
            model = Sequential()
            model.add(tf.keras.Input(shape=self.input_dim))
            if self.layer == 'lstm':
                for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                    model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=self.stepUnit, max_value=self.maxUnits, 
                                          step=self.stepUnit),return_sequences=True))
                model.add(LSTM(hp.Int('layer_2_neurons', min_value=self.stepUnit, max_value=self.maxUnits, 
                                      step=self.stepUnit),return_sequences=False))
            elif self.layer == 'bilstm':
                for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                    model.add(Bidirectional(LSTM(hp.Int(f'lstm_{i}_units', min_value=self.stepUnit, max_value=self.maxUnits, 
                                          step=self.stepUnit),return_sequences=True)))
                model.add(Bidirectional(LSTM(hp.Int('layer_2_neurons', min_value=self.stepUnit, max_value=self.maxUnits, 
                                      step=self.stepUnit),return_sequences=False)))
            if self.layer == 'gru':
                for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                    model.add(GRU(hp.Int(f'lstm_{i}_units', min_value=self.stepUnit, max_value=self.maxUnits, 
                                          step=self.stepUnit),return_sequences=True))
                model.add(GRU(hp.Int('layer_2_neurons', min_value=self.stepUnit, max_value=self.maxUnits, 
                                      step=self.stepUnit),return_sequences=False))
            elif self.layer == 'bigru':
                for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                    model.add(Bidirectional(GRU(hp.Int(f'lstm_{i}_units', min_value=self.stepUnit, max_value=self.maxUnits, 
                                          step=self.stepUnit),return_sequences=True)))
                model.add(Bidirectional(GRU(hp.Int('layer_2_neurons', min_value=self.stepUnit, max_value=self.maxUnits, 
                                      step=self.stepUnit),return_sequences=False)))
            model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=self.maxDropout, step=self.dropoutRateStep)))
            for i in range(hp.Int('n_layers', 1, self.numberOfLayers)):
                model.add(Dense(hp.Int(f'dense_{i}_units', min_value=self.stepDenseUnit, max_value=self.maxDenseUnits, 
                                       step=self.stepDenseUnit)))
            model.add(Dense(self.dense_out, activation='leaky_relu'))
            model.compile(loss=loss_f, optimizer=opt, metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
            return model
        


class ModelTrainer: 
    def __init__(self, config): 
        self.config = config
        self.directory = self.config.root_dir + self.config.experiment_name
    
    def tune_modelClass(self,X, y):
        hypermodel = modelHO_New(self.config) # for class tuner
        tuner = BayesianOptimization (
            hypermodel,
            objective= self.config.objective_metric,
            max_trials=self.config.max_tuner,
            seed=1,
            executions_per_trial=1,
            directory= self.directory,
            project_name=self.config.experiment_name
        )
        es = EarlyStopping(monitor='val_loss', patience=self.config.patience) #What is val_loss? 
        
        mc = ModelCheckpoint(self.directory + '/%s.h5' % self.config.experiment_name, save_best_only=True, monitor='val_loss')
        
        history = tuner.search(x=X,y=y,epochs=self.config.epochs,batch_size=self.config.batch_size,validation_split=self.config.validation_split,callbacks=[es, mc])

        tuner.results_summary()

        best_model = tuner.get_best_models(num_models=1)[0]

        return best_model, history
   
    def transfer_learning(self, X, y, model, technique):
        tf_model = TransferLearningModel()

        if technique == 1:
            model = tf_model.transfer_learning1(X, y, model,self.config)
        elif technique == 2:
            model = tf_model.transfer_learning2(X, y, model,self.config)
        elif technique == 3:
            model = tf_model.transfer_learning3(X, y, model,self.config)
        elif technique == 4:
            model = tf_model.transfer_learning4(X, y, model,self.config)
        elif technique == 5:
            model = tf_model.transfer_learning5(X, y, model,self.config)
        elif technique == 6:
            model = tf_model.transfer_learning6(X, y, model,self.config)
        elif technique == 7:
            model = tf_model.transfer_learning7(X, y, model,self.config)
        elif technique == 8:
            model = tf_model.transfer_learning8(X, y, model,self.config)
        elif technique == 9:
            model = tf_model.transfer_learning9(X, y, model,self.config)
        elif technique == 10:
            model = tf_model.transfer_learning10(X, y, model,self.config)
        elif technique == 11:
            model = tf_model.transfer_learning11(X, y, model,self.config)
        elif technique == 12:
            model = tf_model.transfer_learning12(X, y, model,self.config)
        elif technique == 13:
            model = tf_model.transfer_learning13(X, y, model,self.config)
        elif technique == 14:
            model = tf_model.transfer_learning14(X, y, model,self.config)
        elif technique == 15:
            model = tf_model.transfer_learning15(X, y, model,self.config)
        elif technique == 16:
            model = tf_model.transfer_learning16(X, y, model,self.config)
        elif technique == 17:
            model = tf_model.transfer_learning17(X, y, model,self.config)
        else:
            raise ValueError("Invalid transfer learning technique")

        return model

    def checkTunerClass(self, experiment):
        hypermodel = modelHO_SEGAN_LSTM(self.config) # for class tuner

        tuner = BayesianOptimization (
            hypermodel,
            objective=self.objective_metric,
            max_trials=self.max_tuner,
            seed=1,
            executions_per_trial=1,
            directory= self.directory + '/models/',
            project_name=experiment
        )
        best_model = tuner.get_best_models(num_models=1)[0]

        return best_model




