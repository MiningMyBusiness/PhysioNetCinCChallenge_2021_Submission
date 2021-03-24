import numpy as np
import pandas as pd
import pickle
from skmultilearn.model_selection import iterative_train_test_split
from imblearn.over_sampling import RandomOverSampler
import copy

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, UpSampling1D, AveragePooling1D, Flatten, Reshape, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

import glob

class OVR_DNN:
    
    def __init__(self, X_train=None, y_train=None, filename=None):
        self._X_train = X_train
        self._y_train = y_train
        self._filename = filename
        if self._X_train is not None:
            self.train_base_models()
            self.train_stk_models()
        if self._filename is not None:
            print('Loading models...')
            self.load_models()
        
        
        
    def train_base_models(self):
        self.split_data()
        all_models = []
        # train one model for each class
        for col in range(self._y_train.shape[1]):
            print('Training one vs rest DNN for column:', col+1)
            this_y_train = self._y_train[:,col]
            if len(set(this_y_train)) == 1:
                this_y_train[-1] = 1
            this_y_val = self._y_val[:,col]
            this_model = self.get_model()
            ros = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = ros.fit_resample(self._X_train, this_y_train)
            this_model.compile(optimizer='Adam',
                            loss='binary_crossentropy',
                            metrics=['AUC', 'Precision', 'Recall'])
            my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
            weight_arr = compute_class_weight('balanced', classes=[0,1], y=y_resampled)
            class_weight = {0:weight_arr[0], 1:weight_arr[1]}
            print('  found class weight:', class_weight)
            history = this_model.fit(X_resampled, y_resampled,
                            epochs=16,
                            batch_size=32,
                            shuffle=True,
                            callbacks=my_callbacks,
                            class_weight=class_weight,
                            validation_data=(self._X_val, this_y_val))
            all_models.append(copy.copy(this_model))
        self._dnn_models = all_models
        
        
        
        
    def get_model(self):
        input_vec = Input(shape=(self._X_train.shape[1],))
        x = Dense(self._X_train.shape[1], activation='relu')(input_vec)
        x = Dropout(0.2)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(input_vec, predictions)
        return model
    
    
    def split_data(self):
        print('Splitting data into training and validation set to train DNNs...')
        X_train, y_train, X_val, y_val = iterative_train_test_split(self._X_train, 
                                                              self._y_train, 
                                                              test_size = 0.35)
        self._X_train = X_train
        self._y_train = y_train
        self._X_val = X_val
        self._y_val = y_val
        print('Train data:', X_train.shape)
        print('Train labels:', y_train.shape)
        print('Val data:', X_val.shape)
        print('Val labels:', y_val.shape)
            
            
    def train_stk_models(self):
        print('Training stacking model on validation set...')
        for i,model in enumerate(self._dnn_models):
            print('  Getting probabilities for validation set...')
            this_y_prob = model.predict(self._X_val)
            this_y_prob = this_y_prob.reshape(-1,1)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1) 
                
        stk_models = []
        for i in range(self._y_val.shape[1]):
            print('  Training one vs rest stack model for column:', i+1)
            this_model = LogisticRegression(class_weight='balanced')
            this_y_val = self._y_val[:, i]
            if len(set(this_y_val)) == 1:
                this_y_val[-1] = 1
            this_model = this_model.fit(y_prob, this_y_val)
            stk_models.append(this_model)
        self._stk_models = stk_models
    
    
    
    def predict(self, X_test):
        for i,model in enumerate(self._dnn_models):
            this_y_prob = model.predict(X_test)
            this_y_prob = this_y_prob.reshape(-1,1)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1)        
        
        for i,model in enumerate(self._stk_models):
            this_y_pred = model.predict(y_prob)
            this_y_pred = this_y_pred.reshape(-1,1)
            if i == 0:
                y_pred_test = this_y_pred
            else:
                y_pred_test = np.concatenate((y_pred_test, this_y_pred), axis=1)
        return y_pred_test
    
    
    
    def predict_proba(self, X_test):
        for i,model in enumerate(self._dnn_models):
            this_y_prob = model.predict(X_test)
            this_y_prob = this_y_prob.reshape(-1,1)
            if i == 0:
                y_prob = this_y_prob
            else:
                y_prob = np.concatenate((y_prob, this_y_prob), axis=1)        
        
        for i,model in enumerate(self._stk_models):
            this_y_pred = model.predict_proba(y_prob)
            this_y_pred = this_y_pred[:,1].reshape(-1,1)
            if i == 0:
                y_pred_test = this_y_pred
            else:
                y_pred_test = np.concatenate((y_pred_test, this_y_pred), axis=1)
        return y_pred_test
    
    
    
    def save_models(self, filename):
        # save base dnn models
        for i,model in enumerate(self._dnn_models):
            this_name = filename.split('.pick')[0] + '_basednn' + str(i).zfill(2) + '.h5'
            model.save(this_name)
        # save stack sklearn models
        stk_filename = filename.split('.pick')[0] + '_stkmodels.pickle'
        with open(stk_filename, 'wb') as handle:
            pickle.dump(self._stk_models, handle)
        handle.close()
        
    
    
    
    def load_models(self):
        # load in base dnn models
        file_hash = self._filename.split('.pick')[0] + '_basednn*.h5'
        dnn_files = np.sort(np.array(glob.glob(file_hash)))
        self._dnn_models = []
        for file in dnn_files:
            this_model = tf.keras.models.load_model(file)
            self._dnn_models.append(this_model)
        # load in pickle models
        stk_filename = self._filename.split('.pick')[0] + '_stkmodels.pickle'
        with open(stk_filename, 'rb') as handle:
            self._stk_models = pickle.load(handle)
        handle.close()
    
    
    