import os, random, pywt, sys, pdb, datetime, collections, math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from prettytable import PrettyTable, from_html_one
from sklearn.metrics import roc_curve, auc


import dataset_functions as dfunctions
import signal_analysis_functions as sfunctions
import classifier_functions as cfunctions




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
        data_base_dir = 'dataset\\'
    elif signal_type == "simulated":
        data_base_dir = 'simulated_dataset\\'
    result_base_dir = 'pso_svm_classification_output\\'

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
    classifier_names = ["SVM(RBF)", "SVM(POLY)", "KNN"]
    classifier_objects = [cfunctions.getSVMRBF, cfunctions.getSVMPOL, cfunctions.getKNN]
    classification_feature_functions = [
        sfunctions.calculate_mav_single,
        sfunctions.calculate_avp_single,
        sfunctions.calculate_std_single,
        sfunctions.calculate_ram
    ]
    classification_features = []
    classification_result_path_test = os.path.join(result_base_dir, "average_performance_graph_test" + suffix + ".html")
    classification_result_path_val = os.path.join(result_base_dir,
                                                  "average_performance_graph_validation" + suffix + ".html")
    if os.path.exists(classification_result_path_test):
        with open(classification_result_path_test, 'r') as f:
            classification_result_table_test = from_html_one(f.read())
    else:
        classification_result_table_test = PrettyTable()
        classification_result_table_test.field_names = ["SL No.", "Classifier", "Avg. Test Acc.", "Avg. Test Specificity",
                                                        "Avg. Test Sensitivity"]
    if os.path.exists(classification_result_path_val):
        with open(classification_result_path_val, 'r') as f:
            classification_result_table_val = from_html_one(f.read())
    else:
        classification_result_table_val = PrettyTable()
        classification_result_table_val.field_names = ["SL No.", "Classifier", "Avg. Validation Acc.",
                                                       "Avg. Validation Specificity",
                                                       "Avg. Validation Sensitivity"]
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
    data_prep_verbose = False
    data_prep_plot = False
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
    1.  For each filtered data:
        1.1 For each wavelet sub-band of the filtered data:
            1.1.1 For each feature in classification feature list:
                1.1.1.1 Extract the feature and store it to array containing feature values
            1.1.2 Store the array containing feature values to the list containing features of each wavelet
        1.2 Store the array containing feature of each wavelet as Input feature vector to the classifier.
    """
    feature_extract_verbose = False
    # Extract Features from wavelets of all filtered data
    input_features = [] # Feature Vector for the Classifier
    # For each filtered data
    for i in range(len(data_wavelets)):
        d = data_wavelets[i] # Signal data
        if feature_extract_verbose:
            print("Extracting feature from Signal No. " + str(i+1) + " of shape: " + str(np.asarray(d).shape))
        wavelet_features = [] # Feature Vector for the Signal
        for j in range(len(d)):
            w = d[j]
            if feature_extract_verbose:
                print("....Extracting features from Wavelet No. " + str(j+1) + " of length: " + str(len(w)))
            features = [] # Features for the Wavelet

            mav = sfunctions.calculate_mav_single(w.copy())  # Calculate Mean Absolute Value
            wavelet_features.append(mav) # Store the Features to Wavelet Features array
            if feature_extract_verbose:
                print("........Mean Absolute Value: " + str(mav))
            avp = sfunctions.calculate_avp_single(w.copy())  # Calculate Average Power
            wavelet_features.append(avp) # Store the Features to Wavelet Features array
            if feature_extract_verbose:
                print("........Average Power: " + str(avp))
            std = sfunctions.calculate_std_single(w.copy())  # Calculate Standard Deviation
            wavelet_features.append(std) # Store the Features to Wavelet Features array
            if feature_extract_verbose:
                print("........Standard Deviation: " + str(std))
            # If the current wavelet has a next sibling
            if j < len(d) - 1:
                # Calcuate Ratio of Mean Absolute Value between adjacent wavelets
                ram = sfunctions.calculate_ram_single(w, d[j + 1].copy())
                wavelet_features.append(ram) # Store the Features to Wavelet Features array
                if feature_extract_verbose:
                    print("........Ratio of Mean Absolute Value: " + str(ram))
        if feature_extract_verbose:
            print("....Total Features received: " + str(len(wavelet_features)))
        input_features.append(wavelet_features) # Store Feature array for each signal to the list of input data
    if feature_extract_verbose:
        print("Total Feature Arrays received for input to the classifier: " + str(len(input_features)))

        # ------------------------------3. CLASSIFICATION AND PERFORMANCE-------------------------------------------------
    """
    This section contains the code for classifying the feature data. The Classifiers which will be used for
    classification are:
    1. Support Vector Machine with RBF Activation and Particle Swarm Optimization for adjusting Gamma and Tolerance Parameters.
    2. Support Vector Machine with Polynomial Activation and Particle Swarm Optimization for adjusting Gamma and Tolerance Parameters.
    3. K Nearest Neighbors with Particle Swarm Optimization for adjusting number of neighbors.
    4. Random Forest Algorithm with Particle Swarm Optimization for adjusting number of decision trees.

    The steps followed in order to classify the input feature vector are:
    1.  Create Input Feature vector from the list of feature vectors and their corresponding class labels.
    2. For each Classifier:
        2.1 Create Figure to plot performance.
        2.2 For each data set size:
            2.2.1 Split Train and Validation data set based on 10 fold Corss validation
            2.2.2 Split Train and Test data set based on 10 fold cross validation
            2.2.3 Optimize Classifier using the Train and Test data set
            2.2.4 Classify using Test data set.
            2.2.5 Plot ROC Curve and AUC for the test data set.
            2.2.6 Store Test Accuracy, Specificity and Sensitivity
            2.2.7 Classify using the validation data set.
            2.2.8 Plot ROC Curve and AUC for the validation data set.
            2.2.9 Store Validation Accuracy, Sensitivity and Specificity of the validation data set
        2.3 Plot Train, Test, Specificity and Sensitivity for increasing data size.
        2.4 Add Train Accuracy, Test Accuracy, Sensitivity and Specificity for increasing data set
            to table row .
        2.5 Add Avg Train Accuracy, Average Test Accuracy, Average Specificity, Average Sensitivity for
            increasing data set size to performance output table.
    3. Display Table            
    """

    fig_num=10
    lw = 2
    classification_verbose = True
    classification_plot = True
    # Create Input Vectors for the classifiers
    features = np.asarray(input_features)
    if scale_data:
        features = features/np.amax(np.abs(features))
    # Create Class Labels for the classifiers
    labels = np.asarray(data_labels)
    if classification_verbose:
        print("Input data shape: " + str(features.shape))
        print("Input label shape: " + str(labels.shape))

    # For each classifier
    for i in range(len(classifier_names)):
        if classification_verbose:
            print("Performing Classification using " + classifier_names[i].upper())

        if classification_plot:
            # Plot Figure Title for displaying ROC Curve for the Classifier
            plt.figure(fig_num + i)
            plt.suptitle('ROC - Classifier: ' + str(classifier_names[i].upper()))

        fpr_test = []
        tpr_test = []
        acc_test = []
        inpsize_test = []

        inpsize_val = []
        fpr_val = []
        tpr_val = []
        acc_val = []

        # For each Input Data Size
        for j in range(len(data_size)):
            if classification_verbose:
                print("....Performing Classification for data set of size: " + str(data_size[j]))
            X = features[:data_size[j], :] # Get Input feature
            y = labels[:data_size[j]] # Get Input Label
            if classification_verbose:
                print("....Input Size: " + str(X.shape))
                print("....Label Size: " + str(y.shape))
            # Split train and validation data set
            X_input, X_validate, y_input, y_validate = train_test_split(X, y, test_size=0.1, shuffle=True)
            while len(collections.Counter(list(y_input))) <= 1:
                X_input, X_validate, y_input, y_validate = train_test_split(X, y, test_size=0.1, shuffle=True)
            if classification_verbose:
                print("....Train & Validation data split")
                print("........Train data size: " + str(X_input.shape))
                print("........Validation data size: " + str(X_validate.shape))
                print("........Train label size: " + str(y_input.shape))
                print("........Validation label size: " + str(y_validate.shape))
            # Split Train and Test data set
            X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.1, shuffle=True)
            while len(collections.Counter(list(y_train))) <= 1:
                X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.1, shuffle=True)
            if classification_verbose:
                print("....Train & Test data split")
                print("........Train data size: " + str(X_train.shape))
                print("........Test data size: " + str(X_test.shape))
                print("........Train label size: " + str(y_train.shape))
                print("........Test label size: " + str(y_test.shape))

            # Optimize classifier using train and test data
            classifier, _ = classifier_objects[i](X_input.copy(), y_input.copy())

            # Classify using test data
            classifier.fit(X_train, y_train)
            if classification_verbose:
                print("....Classification accuracy using test data: " + "{0:.2f}".format(classifier.score(X_test, y_test)*100))

            test_probs = classifier.predict_proba(X_test)
            inpsize_test.append(data_size[j])

            # Compute binary class output for test data in order to calculate ROC
            y_test_bin = np.empty((len(y_test), len(label_map)))
            for k in range(y_test_bin.shape[0]):
                arr = [0 for _ in range(len(label_map))]
                arr[y_test[k]] = 1
                y_test_bin[k] = np.asarray(arr)
            if classification_verbose:
                print('....Test Label original shape: ' + str(np.asarray(y_test).shape))
                print('....Test Label binary shape: ' + str(y_test_bin.shape))
                print('....Test score shape:' + str(test_probs.shape))
            # Compute ROC curve and ROC area for Patient class of test data
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for k in range(len(label_map)):
                if k == als_patient_label:
                    fpr[0], tpr[0], _ = roc_curve(y_test_bin[:, k], test_probs[:, k])
                    roc_auc[0] = auc(fpr[0], tpr[0])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), test_probs.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            if classification_plot:
                # Plot ROC Curve for test data classification performance
                plt.subplot(2, 1, 1)
                plt.title("Test data")
                plt.plot(fpr[0], tpr[0], lw=lw,
                         label='ROC curve (area = %0.2f)' % roc_auc[0] + ', Input Size: ' + str(data_size[j]))
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.legend(loc="lower right")
                plt.xlabel('False Positive Rate(1-Specificity)')
                plt.ylabel('True Positive Rate(Sensitivity)')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.grid()

            # Store Test Accuracy, Specificity and Sensitivity
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for k in range(len(y_test)):
                if y_test[k] == als_patient_label:

                    if y_test[k] == np.argmax(test_probs[k]):
                        tp += 1
                    else:
                        fn += 1
                else:
                    if y_test[k] == np.argmax(test_probs[k]):
                        tn += 1
                    else:
                        fp += 1
            if (tp + fn) > 0:
                tpr_test.append(tp / (tp + fn))
            else:
                tpr_test.append(0)
            if (tn + fp) > 0:
                fpr_test.append(tn / (tn + fp))
            else:
                fpr_test.append(0)
            # tpr_test.append(tpr[0])
            # fpr_test.append(fpr[0])
            acc_test.append(classifier.score(X_test, y_test))

            # Optimize classifier using train and validation data
            classifier, _ = classifier_objects[i](X_input.copy(), y_input.copy())
            # Classify using test data
            classifier.fit(X_input, y_input)
            if classification_verbose:
                print("....Classification accuracy using validation data: " + "{0:.2f}".format(
                    classifier.score(X_validate, y_validate) * 100))

            val_probs = classifier.predict_proba(X_validate)
            inpsize_val.append(data_size[j])
            # Compute binary class output for measuring performance of validation data classification
            y_validate_bin = np.empty((len(y_validate), len(label_map)))
            for k in range(y_validate_bin.shape[0]):
                arr = [0 for _ in range(len(label_map))]
                arr[y_validate[k]] = 1
                y_validate_bin[k] = np.asarray(arr)
            if classification_verbose:
                print('....Validation Label original shape: ' + str(np.asarray(y_validate).shape))
                print('....Validation Label binary shape: ' + str(y_validate_bin.shape))
                print('....Validation score shape:' + str(val_probs.shape))
            # Compute ROC curve and ROC area for Patient Class of validation data
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for k in range(len(label_map)):
                if k == als_patient_label:
                    fpr[0], tpr[0], _ = roc_curve(y_validate_bin[:, k], val_probs[:, k])
                    roc_auc[0] = auc(fpr[0], tpr[0])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_validate_bin.ravel(), val_probs.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            if classification_plot:
                plt.subplot(2, 1, 2)
                plt.title("Validation data")
                plt.plot(fpr[0], tpr[0], lw=lw,
                         label='ROC curve (area = %0.2f)' % roc_auc[0] + ', Input Size: ' + str(data_size[j]))
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.legend(loc="lower right")
                plt.xlabel('False Positive Rate(1-Specificity)')
                plt.ylabel('True Positive Rate(Sensitivity)')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.grid()

            # Store Validation Accuracy, Specificity and Sensitivity
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for k in range(len(y_validate)):
                if y_validate[k] == als_patient_label:

                    if y_validate[k] == np.argmax(val_probs[k]):
                        tp += 1
                    else:
                        fn += 1
                else:
                    if y_validate[k] == np.argmax(val_probs[k]):
                        tn += 1
                    else:
                        fp += 1
            if (tp + fn) > 0:
                tpr_val.append(tp / (tp + fn))
            else:
                tpr_val.append(0)
            if (tn + fp) > 0:
                fpr_val.append(tn / (tn + fp))
            else:
                fpr_val.append(0)
            # tpr_test.append(tpr[0])
            # fpr_test.append(fpr[0])
            acc_val.append(classifier.score(X_validate, y_validate))
        if classification_plot:

            # Plot Validation Accuracy, Test Accuracy, Validation Specificity and Test Sensitivity
            plt.figure(fig_num + 10 + i)
            plt.xlabel('No. of Input Data')
            plt.ylabel('Performance')
            plt.suptitle('Performance Graph - Feature: ' + str(classifier_names[i].upper()))
            plt.plot(inpsize_test, np.asarray(acc_test) * 100, label='Test Accuracy')
            plt.plot(inpsize_test, np.asarray(acc_val) * 100, label='Validation Accuracy')
            plt.plot(inpsize_test, np.asarray(tpr_test) * 100, label='Sensitivity')
            plt.plot(inpsize_test, np.asarray(fpr_test) * 100, label='Specificity')
            plt.grid()
            plt.legend(loc='upper left')

        # classification_result_table.field_names = ["SL No.", "Feature", "Avg. Test Acc.", "Avg. Test Specificity",
        #                               "Avg. Test Sensitivity"]
        sl_no = len(classification_result_table_test._rows) + 1
        clf = classifier_names[i].upper()
        avg_test_acc = np.average(np.asarray(acc_test) * 100)
        avg_test_sensitivity = np.average(np.asarray(tpr_test) * 100)
        avg_test_specificity = np.average((1 - np.asarray(fpr_test)) * 100)
        classification_result_table_test.add_row(
            [sl_no, clf, avg_test_acc, avg_test_specificity, avg_test_sensitivity])

        sl_no = len(classification_result_table_val._rows) + 1
        avg_val_acc = np.average(np.asarray(acc_val) * 100)
        avg_val_sensitivity = np.average(np.asarray(tpr_val) * 100)
        avg_val_specificity = np.average(np.asarray(fpr_val) * 100)
        classification_result_table_val.add_row([sl_no, clf, avg_val_acc, avg_val_specificity, avg_val_sensitivity])

    if classification_verbose:
        print("....Classification Performance: Test")
        print(classification_result_table_test.get_string())
        print("....Classification Performance: Validation")
        print(classification_result_table_val.get_string())
    if classification_plot:
        plt.show()