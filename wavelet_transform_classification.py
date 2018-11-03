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
    signal_type = "real"
    scale_data = False
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

    cropped_signal_duration = 5000 #ms

    signal_filter_band = 'band'
    signal_filter_range = [5, 10000] # Highpass: 5Hz, Lowpass: 10KHz
    signal_filter_order = 2

    dwt_mother_wavelet = 'db4'
    dwt_level = 5
    discrete_wavelet_output_table = PrettyTable()
    discrete_wavelet_output_table.field_names = ['SL No.', 'Subject Type']

    classification_feature_labels = ["Mean of Absolute Value", "Average Power",
                                     "Standard Deviation", "Ratio of Absolute Mean Value"]
    classification_features = []
    save_features = True

    # ------------------------------1. DATA ACQUISITION-------------------------------------------------
    """
    This section loads raw signal data from the urls and arranges it in an array for preprocessing.
    The steps followed are:
    1. For each URL:
        1.1 Load Numpy data.
        1.2 Read Sampling rate
        1.3 Pad/Crop raw Input data in order to make all sample data of same length.
        1.4 Store the Cropped data and their corresponding labels
    """
    data_np = []
    data_labels = []
    data_fs = []
    data_acq_verbose = False

    for i in range(len(urls)):
        # Load Numpy data
        d = np.load(os.path.join(urls[i], data_filename))
        # Read Sampling rate
        fs = dfunctions.read_sampling_rate(os.path.join(urls[i], header_filename))


        # Crop data
        cropped_data_length = int( (fs*cropped_signal_duration)/1000 )
        extra = len(d) - cropped_data_length
        if extra > 0:
            crop_left = int(extra/2)
            crop_right = extra - crop_left
            cropped_data = d[crop_left:-crop_right]
        elif extra < 0:
            zeros_left = int(abs(extra)/2)
            zeroes_right = abs(extra) - zeros_left
            cropped_data = np.asarray(
                [0 for _ in range(zeros_left)] + d.tolist() + [0 for _ in range(zeroes_right)]
            )
        else:
            cropped_data = d.copy()

        # Store Cropped data, label and Sampling rate
        data_np.append(cropped_data)
        data_fs.append(fs)
        data_labels.append(labels[i])

        if data_acq_verbose:
            print("Loaded data from: " + urls[i])
            print("Subject Type: " + str(label_map[labels[i]]))
            print("Original Signal duration: " + str( (1000*d.shape[0])/fs ) + "ms")
            print("Cropped Signal duration: " + str((1000 * cropped_data.shape[0]) / fs) + "ms")
            print("----------------------------------------------------------------------------------\n\n\n")

    # ------------------------------2. SIGNAL PREPROCESSING-------------------------------------------------
    """
    This section preprocesses the cropped data by filtering it. The steps followed are:
    
    2. For each cropped data:
        2.1 Butterpass Filter the data with specified filter parameters.     
        2.2 Add Filtered data list to the All Filtered data list
        2.3 Create discrete wavelet transform of the filtered data using specified parameters
        2.4 Store the wavelet coefficients of the Discrete Wavelet Transform
    """
    data_filtered = []
    data_wavelets = []
    data_prep_verbose = True
    data_prep_plot = True
    if data_prep_plot:
        plt.ion()
        plt.figure(1)
        plt.show()
    for i in range(len(data_np)):
        # Butterpass filter with specified parameters
        filtered_data = sfunctions.butter_bandpass_filter(data_np[i].copy(), signal_filter_range,
                                                     signal_filter_band, data_fs[i], order=signal_filter_order)
        # Add filtered data to list of all filtered data
        data_filtered.append(filtered_data)

        # Create DWT of the filtered data using specified parameters
        wavelets = sfunctions.calculate_dwt(filtered_data, method=dwt_mother_wavelet, level=dwt_level, threshold=False)

        # Store Wavelet coefficients
        data_wavelets.append(wavelets)

        if data_prep_verbose:
            print("Loaded data from: " + urls[i])
            print("Subject Type: " + str(label_map[labels[i]]))
            print("Filtered Signal duration: " + str((1000 * filtered_data.shape[0]) / fs) + "ms")
            print('Number of Wavelets: ' + str(len(wavelets)))
            print("----------------------------------------------------------------------------------\n\n\n")
        if data_prep_plot:
            plt.clf()
            plt.suptitle('Signal Preprocessing(Cropping and Bandpass Filtering)')
            cols = 2
            rows = int(math.ceil( (len(wavelets)+1)/2 ))
            plt.subplot(rows, cols, 1)
            plt.title("Filtered data-Len: " + str(len(filtered_data)))
            plt.plot(filtered_data)
            plt.xlabel('Samples[n]')
            plt.ylabel('Amplitude[uV]')
            plt.grid(True)
            plt.subplots_adjust(wspace=0.3, hspace=0.3)

            for j in range(len(wavelets)):
                plt.subplot(rows, cols, j+2)
                plt.title('Dec No. ' + str(j+1))
                plt.plot(wavelets[j])
                plt.grid(True)
                plt.subplots_adjust(wspace=0.3, hspace=0.5)
            plt.pause(2)

    # ------------------------------3. FEATURE EXTRACTION-------------------------------------------------
    """
    This section contains the code for extracting features from the preprocessed data. The Features that will be
    extracted from each frame of each data are as follows:
    1. Frequency Distribution: Mean of Absolute Value of the coefficients of DWT in each sub-band.
    2. Frequency Distribution: Average Power of the coefficients of DWT in each sub-band.
    3. Amount of Change in Frequency Distribution: Standard Deviation of the coefficients of DWT in each sub-band.
    4. Amount of Change in Frequency Distribution: Ratio of Absolute Mean value of adjacent sub-bands.
    
    The steps followed in order to extract feature from each sub-band of Wavelet coefficients of each filtered data
    are as follows:
    1. 
    """


