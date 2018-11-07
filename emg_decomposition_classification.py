import os, random, pywt, sys, pdb, datetime, collections, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram, find_peaks
from sklearn.svm import SVC, SVR
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable, from_csv, from_html_one
from MyNet.emg_classification_library.particle_swarm_optimization import Particle, PSO
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelBinarizer

import MyNet.emg_classification_library.dataset_functions as dfunctions
import MyNet.emg_classification_library.signal_analysis_functions as sfunctions



if __name__ == "__main__":
    # ------------------------------PREDEFINED PARAMETERS-----------------------------------------------
    """
    Predefined parameters for loading the dataset, data preprocessing, feature extraction, classification and
    performance evaluation are set here. The parameters are:

    -------------------- A. Data Acquisition Section----------------------------------------
    1. Signal Type: Selects whether to load the dataset of simulated or real dataset
    2. Scale Data: Selects whether to normaliza input data to the Classifier
    3. Suffix: The name with which numoy arrays of results(etracted features, labels, label map) will be saved
    4. Dataset Base Directory: The root directory for the dataset.
    5. Result Base Directory: The root directory where result outputs will be saved
    6. URLS, Labels and Label Map: The urls of input data, labels of input data and map of the label class names
    7. Patient Label: The class label for which performance(ROC Curve) will be evaluated.

    --------------------- B. PreProcessing Section--------------------------------------------
    1. Crop length: The number of samples which will be cropped for each input data.
    2. Crop Left and Right length: The amout of samples that will be removed from left and right
    3. Filter Band: The Band Type(Lowpass, Highpass, Bandpass) which will be used for filtering the data.
    4. Filter Range: The range(Highpass and/or Lowpass) which will be used for filtering the data
    5. DWT Wavelet: Discrete Wavelet Transform Mother Wavelet
    6. DWT level: Number of levels upto which the signal will be decomposed

    ---------------------- C. Feature Extraction Section--------------------------------------
    1. Feature Table: The table in which the output of feature extraction will be stored(Discrete Wavelet Transform).
    2. Classification Features: The features extracted for classification.
    3. Input Features, label and label map path: The path of file where input features to the classifier will be stored.

    """
    signal_type = "simulated"
    scale_data = True
    suffix = "_" + signal_type
    if scale_data:
        suffix = suffix + "_scaled"
    else:
        suffix = suffix + "_unscaled"
    if signal_type == "real":
        data_size = [40, 60, 90, 120, 150, 170]
    elif signal_type == "simulated":
        data_size = [40, 50, 60, 70, 80, 90]
    if signal_type == "real":
        data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\'
    elif signal_type == "simulated":
        data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\simulated_dataset\\'
    result_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\emg_classification_library\\pso_svm_classification_output\\'

    print("LOADING DATA SET")
    raw_urls, raw_labels, label_map = dfunctions.get_dataset(data_base_dir, shuffle=True)
    raw_urls = list(raw_urls)
    raw_labels = list(raw_labels)
    if raw_labels[0] == raw_labels[1]:
        for i in range(1, len(raw_labels)):
            if raw_labels[i] != raw_labels[0]:
                temp_label = raw_labels[i]
                temp_url = raw_urls[i]
                raw_labels[i] = raw_labels[1]
                raw_urls[i] = raw_urls[1]
                raw_labels[1] = temp_label
                raw_urls[1] = temp_url
                break
    urls = []
    labels = []
    uniform_dataset = False
    uniform_dataset_size = 50

    if "als" in label_map:
        als_patient_label = label_map.index("als")
    elif "neuropathy" in label_map:
        als_patient_label = label_map.index("neuropathy")
    else:
        als_patient_label = -1

    if uniform_dataset:
        added = [0 for _ in range(len(label_map))]
        for i in range(len(raw_urls)):
            if added[raw_labels[i]] < uniform_dataset_size:
                urls.append(raw_urls[i])
                labels.append(raw_labels[i])
                added[raw_labels[i]] += 1
        c = list(zip(urls, labels))
        random.shuffle(c)
        urls, labels = zip(*c)
    else:
        urls = raw_urls
        labels = raw_labels

    data_filename = 'data.npy'
    header_filename = 'data.hea'
    print('Dataset Loaded - Total: ' + str(len(urls)) + ', Output Classes: ' + str(len(label_map)))

    if len(urls) % 10 != 0:
        data_size += [len(urls)]

    cropped_signal_duration = 5000  # ms

    signal_filter_band = 'band'
    signal_filter_range = [5, 10000]  # Highpass: 5Hz, Lowpass: 10KHz
    signal_filter_order = 2