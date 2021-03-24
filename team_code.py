#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
import pandas as pd
from slice_dice import Slice_Dice
from Train_Autoencoder import The_Autoencoder
from Encode_Shape import Encode_Shape
from sklearn.neighbors import KNeighborsClassifier
from OVR_DNN import OVR_DNN
import pickle

twelve_lead_model_filename = '12_lead_model.pickle'
six_lead_model_filename = '6_lead_model.pickle'
three_lead_model_filename = '3_lead_model.pickle'
two_lead_model_filename = '2_lead_model.pickle'

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')
    
    # data extraction parameters
    desire_rate = 50 #Hz, desired sampling rate
    time_window = 5 #sec, window of time of ekg to look in
    overlap = 0.2 # proportion of overlap between time windows
    
    # loop through all recordings
    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        this_header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        this_recording = recording.T # transpose data time is rows and columns are features

        # get labels and frequency
        this_freq = get_frequency(this_header)
        this_labels = get_labels(this_header)
        
        # shape data and labels for single recording
        print('  slicing and dicing data...')
        slicer = Slice_Dice(this_recording, this_freq)
        chunks = slicer.process_chunks(desire_rate, 
                                       time_window, 
                                       overlap, 
                                       this_header)
        chunk_labels = [this_labels]*len(chunks)
        group = [recording_files[i]]*len(chunks)
        print('     found', len(chunks), '5 second windows in this file...')
        
        # combine all recordings
        if i == 0:
            all_chunks = chunks
            all_labels = chunk_labels
            all_groups = group
        else:
            all_chunks.extend(chunks)
            all_labels.extend(chunk_labels)
            all_groups.extend(group)
            
    my_encoder = The_Autoencoder(all_chunks, all_groups, 
                                  encoded_dim=100)

    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    leads = twelve_leads
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    shaper = Encode_Shape(my_encoder, all_chunks, leads, 
                          chunk_labels=all_labels)
    features, labels = shaper.get_shaped_output()
    print('  found features of shape:', features.shape)
    print('  found labels of shape:', labels.shape)

    my_ovr = OVR_DNN(X_train=features, y_train=labels)
    save_model(filename, shaper, my_ovr, leads)

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    leads = six_leads
    filename = os.path.join(model_directory, six_lead_model_filename)

    shaper = Encode_Shape(my_encoder, all_chunks, leads, 
                          chunk_labels=all_labels)
    features, labels = shaper.get_shaped_output()
    print('  found features of shape:', features.shape)
    print('  found labels of shape:', labels.shape)

    my_ovr = OVR_DNN(X_train=features, y_train=labels)
    save_model(filename, shaper, my_ovr, leads)

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    leads = three_leads
    filename = os.path.join(model_directory, three_lead_model_filename)

    shaper = Encode_Shape(my_encoder, all_chunks, leads, 
                          chunk_labels=all_labels)
    features, labels = shaper.get_shaped_output()
    print('  found features of shape:', features.shape)
    print('  found labels of shape:', labels.shape)

    my_ovr = OVR_DNN(X_train=features, y_train=labels)
    save_model(filename, shaper, my_ovr, leads)

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    leads = two_leads
    filename = os.path.join(model_directory, two_lead_model_filename)

    shaper = Encode_Shape(my_encoder, all_chunks, leads, 
                          chunk_labels=all_labels)
    features, labels = shaper.get_shaped_output()
    print('  found features of shape:', features.shape)
    print('  found labels of shape:', labels.shape)

    my_ovr = OVR_DNN(X_train=features, y_train=labels)
    save_model(filename, shaper, my_ovr, leads)

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, shaper_object, my_ovr, leads):
    # save encoder
    filename_tf = filename.split('.pick')[0] + '_encoder.h5'
    shaper_object._encoder._encoder.save(filename_tf)
    # save ovr model files
    my_ovr.save_models(filename)
    # save leads
    model_dict = {
        'leads': leads,
    }
    with open(filename, 'wb') as handle:
        pickle.dump(model_dict, handle)
    handle.close()
    

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    # load encoder
    filename_tf = filename.split('.pick')[0] + '_encoder.h5'
    encoder_object = The_Autoencoder(encoder_filename=filename_tf)
    # load ovr models
    my_ovr = OVR_DNN(filename=filename)
    # load leads
    with open(filename, 'rb') as handle:
        saved_dict = pickle.load(handle)
    handle.close()
    model_dict = {
        'encoder': encoder_object,
        'leads': saved_dict['leads'],
        'classifier': my_ovr
    }
    return model_dict

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    leads = model['leads']
    classifier = model['classifier']
    my_encoder = model['encoder']
    
    # Load features
    desire_rate = 50 #Hz, desired sampling rate
    time_window = 5 #sec, window of time of ekg to look in
    overlap = 0.95 # proportion of overlap between time windows
    this_recording = recording.T
    this_freq = get_frequency(header)
    slicer = Slice_Dice(this_recording, this_freq)
    chunks = slicer.process_chunks(desire_rate, 
                                       time_window, 
                                       overlap, 
                                       header)
    shaper = Encode_Shape(my_encoder, chunks, leads, test=True)
    features = shaper.get_shaped_output()

    # Predict labels and probabilities
    many_predicts = classifier.predict(features)
    probabilities = np.mean(many_predicts, axis=0)
    print(probabilities)
    
    labels = np.zeros(len(probabilities)).astype(np.int)
    labels[probabilities >= 0.5] = 1
    print(labels)
    
    classes = shaper._scored_labels
    print(classes)
    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = -1
    age = age/200

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = 0.5

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms
