import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, UpSampling1D, AveragePooling1D, Flatten, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow.keras
import tensorflow as tf

class The_Autoencoder:

    def __init__(self, chunks_list=None, group_list=None, encoded_dim=100,
                encoder_filename=None):
        if chunks_list is not None:
            self._chunk_arr = np.array(chunks_list)
            self._all_groups = np.array(group_list)
            self._encoded_dim = encoded_dim
            self.break_data()
            self.build_autoencoder()
            self.train_autoencoder()
        if encoder_filename is not None:
            self._encoder = tf.keras.models.load_model(encoder_filename)
        
        
        
        
    def expand_chunks(self, chunk_arr):
        break_list = []
        for chunk in chunk_arr:
            for i in range(chunk.shape[1]):
                this_piece = chunk[:,i]
                this_piece = (this_piece - np.min(this_piece))/((np.max(this_piece) - np.min(this_piece)) + 0.000001)
                break_list.append(this_piece)
        break_list = np.array(break_list)
        return break_list
    
    
    
    def break_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self._chunk_arr, self._all_groups, 
                 test_size=0.2)
        self._X_train = self.expand_chunks(X_train)
        self._X_test = self.expand_chunks(X_test)
        
        
    def build_autoencoder(self):
        # input placeholder
        input_img = Input(shape=(self._X_train.shape[1],))
        encoded = Dense(self._encoded_dim, activation='relu')(input_img)
        decoded = Dense(self._X_train.shape[1], activation='sigmoid')(encoded)

        # create seperate encoder
        self._encoder = Model(input_img, encoded)

        # full autoeconder model
        self._autoencoder = Model(input_img, decoded)

        # configure model to be trained
        # per-pixel binary crossentropy loss
        self._autoencoder.compile(optimizer='Adam', loss='mean_squared_error')
        
    
    def train_autoencoder(self):
        my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
        self._history = self._autoencoder.fit(self._X_train, self._X_train,
                epochs=16,
                batch_size=32,
                shuffle=True,
                callbacks=my_callbacks,
                validation_data=(self._X_test, self._X_test))
        
        
    
    def encode(self, chunk_arr):
        chunk_to_rows = self.expand_chunks(chunk_arr)
        chunk_encode = self._encoder.predict(chunk_to_rows)
        chunk_encode_row = self.form_rows(chunk_encode, chunk_arr[0].shape[1])
        return chunk_encode_row
        
        
        
    
    def form_rows(self, row_arr, col_size):
        concat_rows = []
        for i in range(0, len(row_arr) - col_size + 1, col_size):
            this_concat = np.zeros(col_size*row_arr.shape[1])
            for j in range(0, col_size):
                this_concat[j*row_arr.shape[1]:(j+1)*row_arr.shape[1]] = row_arr[i+j,:]
            concat_rows.append(this_concat)
        concat_rows = np.array(concat_rows)
        return concat_rows
                      
                      
                      
    
                      
                      
    