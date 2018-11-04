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







def knn_optimize(x, args):
    classifier = KNeighborsClassifier(n_neighbors=int(x[0]))
    if len(args) == 2:
        X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])

# Test Section Specific modules

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
    1. Data frame length: The number of frames to which the input data will be segmented.
    2. Samples per Frame: Number of sample points per frame of input data.
    3. Crop length: The number of frames which will be cropped.
    4. Crop Start Position: The frame from which cropping will start.
    5. Filter Band: The Band Type(Lowpass, Highpass, Bandpass) which will be used for filtering the data.
    6. Filter Range: The range(Highpass and/or Lowpass) which will be used for filtering the data
    
    
    ---------------------- C. Feature Extraction Section--------------------------------------
    1. Feature Table: The table in which the output of feature extraction will be stored(Average Amplitude).
    2. Classification Features: The features extracted for classification.
    3. Input Features, label and label map path: The path of file where input features to the classifier will be stored.
    
    
    --------------------- D. Classification Section-------------------------------------------
    1. Input Data Sizes: The list of Input Data sizes which will be used for classification of data in each iteration.
    2. Neighbor Range: The range of number of neighbors which will be used for Particle Swarm Optimization(PSO).
    
    ----------------------E. Performance Section-----------------------------------------------
    1. Classification Test result Path: the file path where classification test performance metrics(accuracy,
       sensitivity and specificity) will be stored.
    2. Classification Validation result Path: the file path where classification test performance metrics(accuracy,
       sensitivity and specificity) will be stored.   
    
    """
    signal_type = "real"
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
    result_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\emg_classification_library\\time_freq_classification_output\\'


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


    output_dir = "time_freq_classification_output"
    total_frames = 64
    total_samples_per_frame = 4096
    crop_start = 30
    crop_length = 25
    filter_band = 'lowpass'
    filter_range = [1500]
    avg_amplitude_table = PrettyTable()
    avg_amplitude_table.field_names = ["SL No.", "Subject Type", "Maximum Amplitude",
                                       "Minimum Amplitude", "Average Amplitude",
                                       "Maximum Frequency", "Minimum Frequency",
                                       "Average Frequency"]
    total_als = 3
    total_normal = 3




    total_iterations = len(data_size)
    classifier_neighbor_range = [1, 5]
    spectral_peak_table_path = os.path.join(result_base_dir, "simulated_signal_pso_knn_spectral_peaks_table.html")
    performance_table_path = result_base_dir + "simulated_signal_pso_knn_average_performance_graph.html"
    n_neighbors = 1

    classification_feature_labels = ["Average Spectral Amplitude", "Mean Frequency",
                                     "Zero Lag", "Zero Crossing rate"]
    classification_features = []
    save_features = True

    classification_result_path_test = os.path.join(result_base_dir, "average_performance_graph_test" + suffix + ".html")
    classification_result_path_val = os.path.join(result_base_dir, "average_performance_graph_validation" + suffix + ".html")
    if os.path.exists(classification_result_path_test):
        with open(classification_result_path_test, 'r') as f:
            classification_result_table_test = from_html_one(f.read())
    else:
        classification_result_table_test = PrettyTable()
        classification_result_table_test.field_names = ["SL No.", "Feature", "Avg. Test Acc.", "Avg. Test Specificity",
                                       "Avg. Test Sensitivity"]
    if os.path.exists(classification_result_path_val):
        with open(classification_result_path_val, 'r') as f:
            classification_result_table_val = from_html_one(f.read())
    else:
        classification_result_table_val = PrettyTable()
        classification_result_table_val.field_names = ["SL No.", "Feature", "Avg. Validation Acc.", "Avg. Validation Specificity",
                                       "Avg. Validation Sensitivity"]

    f_file_suffix = suffix + ".npy"
    f_file = os.path.join(result_base_dir,
                          'features_' +  f_file_suffix)
    f_label_file = os.path.join(result_base_dir, 'label_')

    # ------------------------------1. DATA ACQUISITION-------------------------------------------------
    """
    This section loads raw signal data from the urls and arranges it in an array for preprocessing.
    The steps followed are:
    1. For each URL:
        1.1 Load Numpy data.
        1.2 Read Sampling rate
        1.3 Pad/Crop raw Input data in order to make all sample data of same length.
        1.4 Segment the data into specified number of frames where each frame contains specified number of sample points.
        1.5 Store Segmented data, Sampling rate, Label Class and Raw Signal data.
    """

    if os.path.exists(os.path.join(result_base_dir, 'label_'+ f_file_suffix)):
        for i in range(len(classification_feature_labels)):
            classification_features.append(np.load(os.path.join(result_base_dir,
                                        'features_'+classification_feature_labels[i].replace(" ", "")+ f_file_suffix),
                                                   allow_pickle=True))
        labels = list(np.load(os.path.join(result_base_dir,
                                        'label_'+ f_file_suffix),
                                                   allow_pickle=True))
        label_map = list(np.load(os.path.join(result_base_dir,
                                        'label_map_'+ f_file_suffix),
                                                   allow_pickle=True))
    else:

        data_np = []
        segmented_data = []
        sampling_rates = []
        plot_als = []
        plot_normal = []
        for i in range(len(urls)):
            d = np.load(os.path.join(urls[i], data_filename))
            fs = dfunctions.read_sampling_rate(os.path.join(urls[i], header_filename))
            sampling_rates.append(fs)

            # Adjust From both end: Crop data if the total number of sample is larger than expected sample and add zero otherwise
            expected_samples = total_frames * total_samples_per_frame
            extra = len(d) - expected_samples
            left = int(abs(extra) / 2)
            right = abs(extra) - left
            if extra > 0:
                d = d[left:len(d) - right]
            elif extra < 0:
                d = np.asarray([0] * left + list(d.flatten()) + [0] * right)

            # Segment data into specified number of frames and samples per frame
            sd = [d[i:i + total_samples_per_frame] for i in range(0, d.shape[0], total_samples_per_frame)]
            segmented_data.append(sd)

            if labels[i] == als_patient_label and len(plot_als) == 0:
                print("Found als")
                plot_als = [sd, fs, "Neuropathy(Amyotrophic Lateral Sclerosis)", d]
            elif labels[i] != als_patient_label and len(plot_normal) == 0:
                print("Found other")
                plot_normal = [sd, fs, "Healthy Subject", d]

        """if len(plot_als) > 0 and len(plot_normal) > 0:
            plt.figure(1)
    
            signal = []
            for i in range(len(plot_normal[0])):
                signal = signal + list(plot_normal[0][i])
            print("Reconstructed Signal length: " + str(len(signal)))
            print("Original Signal length: " + str(len(plot_normal[3])))
            plt.suptitle("Raw EMG Signal(Biceps Brachii)")
            plt.subplot(2, 2, 1)
            plt.title(plot_normal[2] + "- Sampling rate: " + str(plot_normal[1]) + "Hz")
            plt.plot(signal)
            plt.grid()
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
    
            plt.subplot(2, 2, 2)
            plt.title(plot_normal[2] + "-Fast Fourier Transform")
            fourier = abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1 / plot_normal[1])
            plt.plot(freqs[0:len(freqs)//2], fourier[0:len(freqs)//2])
            plt.grid()
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude of Magnitude Spectrum")
    
    
            signal = []
            for i in range(len(plot_als[0])):
                signal = signal + list(plot_als[0][i])
            print("Reconstructed Signal length: " + str(len(signal)))
            print("Original Signal length: " + str(len(plot_als[3])))
            plt.subplot(2, 2, 3)
            plt.title(plot_als[2] + "- Sampling rate: " + str(plot_als[1]) + "Hz")
            plt.plot(signal)
            plt.grid()
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
    
            plt.subplot(2, 2, 4)
            plt.title(plot_als[2] + "-Fast Fourier Transform")
            fourier = abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1/plot_als[1])
            plt.plot(freqs[0:len(freqs)//2], fourier[0:len(freqs)//2])
            plt.grid()
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude of Magnitude Spectrum")"""

        # ------------------------------2. SIGNAL PREPROCESSING-------------------------------------------------
        """
        This section preprocesses the segmented data by croppong and filtering it. The steps followed are:
        1. For each Segmented data:
            1.1 Crop the segmented data by keeping specified number of frames starting from specified position.
            1.2 Store Cropped data, Sampling rate, Label Class and Segmented Signal data
            
        2. For each cropped data:
            2.1 For each frame of the cropped data:
                2.1.1 Butterpass Filter the frame with specified filter parameters.
                2.1.2 Add back the filtered frame to filtered data list
            2.2 Add Filtered data list to the All Filtered data list
            
        """

        cropped_data = []

        plot_als = []
        plot_normal = []
        for i in range(len(segmented_data)):
            cd = segmented_data[i][crop_start:crop_start+crop_length]
            cropped_data.append(cd)
            if labels[i] == als_patient_label and len(plot_als) == 0:
                print("Found als")
                plot_als = [cd, sampling_rates[i], "Neuropathy(Amyotrophic Lateral Sclerosis)", d]
            elif labels[i] != als_patient_label and len(plot_normal) == 0:
                print("Found other")
                plot_normal = [cd, sampling_rates[i], "Healthy Subject", segmented_data[i]]

        """if len(plot_als) > 0 and len(plot_normal) > 0:
            plt.figure(2)
    
            signal = []
            for i in range(len(plot_normal[0])):
                signal = signal + list(plot_normal[0][i])
            print("Reconstructed Signal length: " + str(len(signal)))
            print("Original Signal length: " + str(len(plot_normal[3])))
            plt.suptitle("Cropped EMG Signal(Biceps Brachii)")
            plt.subplot(2, 2, 1)
            plt.title(plot_normal[2] + "- Sampling rate: " + str(plot_normal[1]) + "Hz")
            plt.plot(signal)
            plt.grid()
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
    
            plt.subplot(2, 2, 2)
            plt.title(plot_normal[2] + "-Fast Fourier Transform")
            fourier = abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1 / plot_normal[1])
            plt.plot(freqs[0:len(freqs)//2], fourier[0:len(freqs)//2])
            plt.grid()
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude of Magnitude Spectrum")
    
            signal = []
            for i in range(len(plot_als[0])):
                signal = signal + list(plot_als[0][i])
            print("Reconstructed Signal length: " + str(len(signal)))
            print("Original Signal length: " + str(len(plot_als[3])))
            plt.subplot(2, 2, 3)
            plt.title(plot_als[2] + "- Sampling rate: " + str(plot_als[1]) + "Hz")
            plt.plot(signal)
            plt.grid()
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
    
            plt.subplot(2, 2, 4)
            plt.title(plot_als[2] + "-Fast Fourier Transform")
            fourier = abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1 / plot_als[1])
            plt.plot(freqs[0:len(freqs)//2], fourier[0:len(freqs)//2])
            plt.grid()
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude of Magnitude Spectrum")"""


        filtered_data = []
        segmented_filtered_data = []
        plot_als = []
        plot_als_b = []
        plot_normal = []
        plot_normal_b = []
        for i in range(len(cropped_data)):
            signal = []
            segmented_signal = []
            for j in range(len(cropped_data[i])):
                signal = signal + list(cropped_data[i][j])
                segmented_signal.append(sfunctions.butter_bandpass_filter(cropped_data[i][j],
                                        filter_range, filter_band, sampling_rates[i], order=2))
            fd = sfunctions.butter_bandpass_filter(np.asarray(signal), filter_range, filter_band, sampling_rates[i], order=2)
            sfd = [fd[i:i + total_samples_per_frame] for i in range(0, fd.shape[0], total_samples_per_frame)]
            filtered_data.append(sfd)
            segmented_filtered_data.append(segmented_signal)

            if labels[i] == als_patient_label and len(plot_als) == 0:
                print("Found als")
                plot_als = [sfd, sampling_rates[i], "Neuropathy(Amyotrophic Lateral Sclerosis)", d]

            elif labels[i] != als_patient_label and len(plot_normal) == 0:
                print("Found other")
                plot_normal = [sfd, sampling_rates[i], "Healthy Subject", d]

            if labels[i] == als_patient_label and len(plot_als_b) == 0:
                print("Found als")
                plot_als_b = [segmented_signal, sampling_rates[i], "Neuropathy(Amyotrophic Lateral Sclerosis)", d]
            elif labels[i] != als_patient_label and len(plot_normal_b) == 0:
                print("Found other")
                plot_normal_b = [segmented_signal, sampling_rates[i], "Healthy Subject", d]

        """if len(plot_als) > 0 and len(plot_normal) > 0:
            plt.figure(3)
    
            signal = []
            for i in range(len(plot_normal[0])):
                signal = signal + list(plot_normal[0][i])
            print("Reconstructed Signal length: " + str(len(signal)))
            print("Original Signal length: " + str(len(plot_normal[3])))
            plt.suptitle("Filtered EMG Signal(Biceps Brachii)")
            plt.subplot(2, 2, 1)
            plt.title(plot_normal[2] + "- Sampling rate: " + str(plot_normal[1]) + "Hz")
            plt.plot(signal)
            plt.grid()
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
    
            plt.subplot(2, 2, 2)
            plt.title(plot_normal[2] + "-Fast Fourier Transform")
            fourier = abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1 / plot_normal[1])
            plt.plot(freqs[0:len(freqs)//2], fourier[0:len(freqs)//2])
            plt.grid()
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude of Magnitude Spectrum")
    
            signal = []
            for i in range(len(plot_als[0])):
                signal = signal + list(plot_als[0][i])
            print("Reconstructed Signal length: " + str(len(signal)))
            print("Original Signal length: " + str(len(plot_als[3])))
            plt.subplot(2, 2, 3)
            plt.title(plot_als[2] + "- Sampling rate: " + str(plot_als[1]) + "Hz")
            plt.plot(signal)
            plt.grid()
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
    
            plt.subplot(2, 2, 4)
            plt.title(plot_als[2] + "-Fast Fourier Transform")
            fourier = abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1 / plot_als[1])
            plt.plot(freqs[0:len(freqs)//2], fourier[0:len(freqs)//2])
            plt.grid()
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude of Magnitude Spectrum")"""


        """if len(plot_als_b) > 0 and len(plot_normal_b) > 0:
            plt.figure(4)
    
            signal = []
            for i in range(len(plot_normal_b[0])):
                signal = signal + list(plot_normal_b[0][i])
            print("Reconstructed Signal length: " + str(len(signal)))
            print("Original Signal length: " + str(len(plot_normal_b[3])))
            plt.suptitle("Segmented Filtered EMG Signal(Biceps Brachii)")
            plt.subplot(2, 2, 1)
            plt.title(plot_normal_b[2] + "- Sampling rate: " + str(plot_normal_b[1]) + "Hz")
            plt.plot(signal)
            plt.grid()
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
    
            plt.subplot(2, 2, 2)
            plt.title(plot_normal_b[2] + "-Fast Fourier Transform")
            fourier = abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1 / plot_normal_b[1])
            plt.plot(freqs[0:len(freqs)//2], fourier[0:len(freqs)//2])
            plt.grid()
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude of Magnitude Spectrum")
    
            signal = []
            for i in range(len(plot_als_b[0])):
                signal = signal + list(plot_als_b[0][i])
            print("Reconstructed Signal length: " + str(len(signal)))
            print("Original Signal length: " + str(len(plot_als_b[3])))
            plt.subplot(2, 2, 3)
            plt.title(plot_als_b[2] + "- Sampling rate: " + str(plot_als_b[1]) + "Hz")
            plt.plot(signal)
            plt.grid()
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
    
            plt.subplot(2, 2, 4)
            plt.title(plot_als_b[2] + "-Fast Fourier Transform")
            fourier = abs(np.fft.fft(signal))
            freqs = np.fft.fftfreq(len(signal), 1 / plot_als_b[1])
    
            plt.plot(freqs[0:len(freqs)//2], fourier[0:len(freqs)//2])
            plt.grid()
            plt.xlabel("Frequency(Hz)")
            plt.ylabel("Amplitude of Magnitude Spectrum")
            plt.show()
    
    
            # Outputs for Signal Preprocessing
    
            # 1. FFT of 5 arbitary frames of each ALS Patients
            random_frames = 5
            total_signals = 1
            remaining_data = [total_signals for _ in range(len(label_map))]
            collected_data = [[] for _ in range(len(label_map))]
            for i in range(len(urls)):
                if remaining_data[labels[i]] > 0:
                    collected_data[labels[i]].append(segmented_filtered_data[i])
                    remaining_data[labels[i]] -= 1
    
            top = ""
            bottom = ""
            if label_map[0] == als_patient_label:
                top = "Neurogenic(ALS)"
                bottom = "Healthy"
            else:
                bottom = "Neurogenic(ALS)"
                top = "Healthy"
            plt.figure(5)
            plt.suptitle("FFT of 5 random frames from each subject: " + top + "-Top, " + bottom + "-Bottom")
            plt.figure(6)
            plt.suptitle("Filtered Signal [" + str(filter_band.upper()) + ": " + str(filter_range[-1]) + " Hz] of 5 random frames from each subject: " + top + "-Top, " + bottom + "-Bottom")
            plt.figure(7)
            plt.suptitle("Autocorrelation of 5 random frames from each subject: " + top + "-Top, " + bottom + "-Bottom")
            current = 1
            for i in range(len(label_map)):
    
                for j in range(total_signals):
                    segments = np.random.randint(0, len(collected_data[i][j]), random_frames)
                    for k in range(random_frames):
                        fourier = np.fft.fft(collected_data[i][j][segments[k]])
                        freqs = np.fft.fftfreq(len(collected_data[i][j][segments[k]]), 1 / filter_range[-1])
                        autocorrelation = sfunctions.calculate_autocorrelation(collected_data[i][j][segments[k]])
    
                        plt.figure(5)
                        plt.subplot(2, random_frames, current)
                        plt.xlabel("Frequency(Hz)")
                        plt.ylabel("Amplitude")
                        plt.grid()
                        plt.title("Segment No. " + str(segments[k]+1))
                        plt.plot(freqs[0 : len(freqs) // 2], abs(fourier[0 : len(fourier) // 2]))
    
    
                        plt.figure(6)
                        plt.subplot(2, random_frames, current)
                        plt.xlabel("Samples[n]")
                        plt.ylabel("Amplitude[uV]")
                        plt.grid()
                        plt.title("Segment No. " + str(segments[k] + 1))
                        plt.plot(collected_data[i][j][segments[k]])
    
    
                        plt.figure(7)
                        plt.subplot(2, random_frames, current)
                        plt.xlabel("Samples[n]")
                        plt.ylabel("Amplitude")
                        plt.grid()
                        plt.title("Segment No. " + str(segments[k] + 1))
                        plt.plot(autocorrelation)
    
    
                        current += 1
            plt.show()"""


        # ------------------------------3. FEATURE EXTRACTION-------------------------------------------------
        """
        This section contains the code for extracting features from the preprocessed data. The Features that will be
        extracted from each frame of each data are as follows:
        1. Frequency Domain: Average Amplitude of the Spectral Peaks obtained from Magnitude Spectrum of the Fast
           Fourier Transform.
        2. Frequency Domain: Mean Frequency of the data obtained via Fast Fourier Transform.
        3. Time Domain: Zero Lag of Autocorrelation obtained from Time series.
        4. Time Domain: Zero Crossing rate obtained from the Time Series.
        
        The steps followed in feature extraction are as follows:
        1. For each filtered data:
            1.1 For each frame of the filtered data:
                1.1.1 Calculate Spectral Peaks for each frame
                1.1.2 Calculate Average Amplitude of the Spectral Peaks from each frame.
                1.1.3 Add the Average Amplitude to the list of feature for each frame of the filtered data.
            1.2 Add the List of feature of the filtered data to the list of Average Amplitude of Spectral Peaks
                for all input data.
        2. Add the Maximum, Minimum Amplitude of Spectral Peaks and their respective frequencies to the feature table.
        3. For each filtered data:
            3.1 For each frame of the filtered data:
                3.1.1 Calculate Mean Frequency of the frame and store it to the list of mean frequency feature of the
                    filtered data.
                3.1.2 Calculate Zero Lag of Autocorrelation of the frame and store it to the list of Zero Lag
                    feature of the filtered data.
                3.1.3 Calculate Zero Crossing rate of the frame and store it to the list of Zero Crossing rate feature
                    of the filtered data.            
                
        4. If Feature Save option is enabled, then save the obtained features along with their label classes and label
           maps to a numpy array.        
        """

        # 1. Average Spectral Amplitude


        avg_spectral_amplitudes = []
        segmented_avg_spectral_amplitudes = []
        for i in range(len(segmented_filtered_data)):
            current = len(avg_amplitude_table._rows)
            max_amp = -sys.maxsize
            min_amp = sys.maxsize
            max_freq = 0
            min_freq = 0
            max_segment = 0
            min_segment = 0
            avg_amp = 0
            avg_freq = 0
            total = 0
            segmented_amps = []
            for j in range(len(segmented_filtered_data[i])):
                fourier, peaks = sfunctions.calculate_spectral_peak(segmented_filtered_data[i][j])
                freqs = np.fft.fftfreq(len(segmented_filtered_data[i][j]), 1/filter_range[-1])
                freqs = freqs[0:len(freqs) // 2]

                for k in range(len(peaks)):
                    if fourier[peaks[k]] > max_amp:
                        max_amp = fourier[peaks[k]]
                        max_freq = freqs[peaks[k]]
                        max_segment = j
                    elif fourier[peaks[k]] < min_amp:
                        min_amp = fourier[peaks[k]]
                        min_freq = freqs[peaks[k]]
                        min_seg = j
                    avg_amp += fourier[peaks[k]]
                    avg_freq += freqs[peaks[k]]
                    total += 1
                segmented_amps.append(np.mean(fourier[peaks]))

            avg_amp = avg_amp/total
            avg_spectral_amplitudes.append(avg_amp)
            avg_freq = avg_freq/total
            segmented_avg_spectral_amplitudes.append(segmented_amps)

            if label_map[labels[i]] == label_map[als_patient_label] and total_als > 0:
                subject_type = "Neurogenic(Amyotrophic Lateral Sclerosis)"
                avg_amplitude_table.add_row([current, subject_type, max_amp, min_amp, avg_amp,
                                             max_freq, min_freq, avg_freq])
                total_als -= 1
            elif label_map[labels[i]] != label_map[als_patient_label] and total_normal > 0:
                subject_type = "Healthy"
                total_normal -= 1
                avg_amplitude_table.add_row([current, subject_type, max_amp, min_amp, avg_amp,
                                             max_freq, min_freq, avg_freq])


        print(avg_amplitude_table.get_string())

        #with open(spectral_peak_table_path, 'w') as fp:
         #   fp.write(avg_amplitude_table.get_html_string())

        # Mean Frequency, zero lag of autocorrelation and Zero Crossing rate
        mean_frequencies = []
        zero_lag = []
        zero_crossing = []
        for i in range(len(segmented_filtered_data)):
            print('Extracting Feature from data no. ' + str(i+1))
            d = segmented_filtered_data[i]
            mf = []
            zl = []
            zc = []
            for j in range(len(d)):
                mf.append(sfunctions.calculate_mean_frequency(d[j], sampling_rates[i]))
                zl.append(sfunctions.calculate_zero_lag_autocorrelation(d[j]))
                zc.append(sfunctions.calculate_zero_crossing_rate(d[j]))
            mean_frequencies.append(mf)
            zero_lag.append(zl)
            zero_crossing.append(zc)
        print('Total data: ' + str(len(urls)))
        print('Labels: ' + str(labels))
        print('Total Mean frequency data: ' + str(len(mean_frequencies)))
        print(mean_frequencies)

        print('Total Zero Lag data: ' + str(len(zero_lag)))
        print(zero_lag)

        print('Total Zero Crossing data: ' + str(len(zero_crossing)))
        print(zero_crossing)

        #feature_result_paths = [result_base_dir + 'features_' + str(classification_feature_labels[i]).replace(" ",
         #                                                                                                    "") + f_file_suffix for i in range(len(classification_feature_labels))]
        #feature_label_paths = [result_base_dir + 'labels_' + str(classification_feature_labels[i]).replace(" ",
         #                                                                                                    "") + f_file_suffix for i in range(len(classification_feature_labels))]
        #label_map_path = result_base_dir + 'label_map_' + str(classification_feature_labels[i]).replace(" ",
          #                                                                                                   "") + f_file_suffix
        classification_features = [segmented_avg_spectral_amplitudes, mean_frequencies,
                                   zero_lag, zero_crossing]

        if save_features:
            for i in range(len(classification_feature_labels)):
                np.save(result_base_dir + 'features_' + str(classification_feature_labels[i]).replace(" ", "")+f_file_suffix,
                        np.asarray(classification_features[i])
                        )
            np.save(os.path.join(result_base_dir,
                                        'label_'+ f_file_suffix),
                                                       np.asarray(labels))
            np.save(os.path.join(result_base_dir,
                                        'label_map_'+ f_file_suffix),
                                                       np.asarray(label_map))
    # ------------------------------4. CLASSIFICATION-------------------------------------------------

    """
    Steps of Classification:
    1. For each feature:
        1.1 Create Figure to plot performance
        1.2 For each data set size:
            1.2.1 Split Train and Validation data set based on 10 fold Corss validation
            1.2.2 Split Train and Test data set based on 10 fold cross validation
            1.2.3 Optimize Classifier using the Train and Test data set
            1.2.4 Classify using Test data set
            1.2.5 Plot ROC Curve and AUC for the test data set
            1.2.6 Store Test Accuracy, Specificity and Sensitivity
            1.2.7 Classify using the validation data set
            1.2.8 Plot ROC Curve and AUC for the validation data set
            1.2.9 Store Validation Accuracy, Sensitivity and Specificity of the validation data set
        1.3 Plot Train, Test, Specificity and Sensitivity for increasing data size
        1.4 Add Train Accuracy, Test Accuracy, Sensitivity and Specificity for increasing data set
            to table row
        1.5 Add Avg Train Accuracy, Average Test Accuracy, Average Specificity, Average Sensitivity for
            increasing data set size to performance output table    
    2. Display Table
    """
    fig_num=10
    for i in range(len(classification_feature_labels)):
        features = classification_features[i]
        if scale_data:
            features = preprocessing.scale(features)
        plt.figure(fig_num+i)
        lw = 2

        plt.suptitle('ROC - Feature: ' + str(classification_feature_labels[i].upper()))

        fpr_test = []
        tpr_test = []
        acc_test = []
        inpsize_test = []

        inpsize_val = []
        fpr_val = []
        tpr_val = []
        acc_val = []

        for j in range(len(data_size)):
            X = np.asarray(features)[:data_size[j], :]
            y = np.asarray(labels)[:data_size[j]]
            # Split train and validation data set
            X_input, X_validate, y_input, y_validate = train_test_split(X, y, test_size=0.1, shuffle=True)
            # Split Train and Test data set
            X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.1, shuffle=True)
            # Optimize classifier using train and test data
            pso = PSO(knn_optimize, [classifier_neighbor_range[1]], [classifier_neighbor_range[0]],
                      fitness_minimize=False, cost_function_args=(X_input, y_input),
                      verbose=False, ndview=False, max_iteration=50)
            knn_particles, knn_global_best, knn_best_costs = pso.run()

            # Classify using test data set
            classifier = KNeighborsClassifier(n_neighbors=int(knn_global_best["position"][0]))
            classifier.fit(X_train, y_train)
            test_probs = classifier.predict_proba(X_test)
            inpsize_test.append(data_size[j])


            # Compute ROC curve and ROC area for each class of test data
            y_test_bin = np.empty((len(y_test), len(label_map)))
            for k in range(y_test_bin.shape[0]):
                arr = [0 for _ in range(len(label_map))]
                arr[labels[k]] = 1
                y_test_bin[k] = np.asarray(arr)
            print('Test Label original shape: ' + str(np.asarray(y_test).shape))
            print('Test Label binary shape: ' + str(y_test_bin.shape))
            print('Test score shape:' + str(test_probs.shape))
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
            plt.subplot(2, 1, 1)
            plt.title("Test data")
            plt.plot(fpr[0], tpr[0], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0] + ', Input Size: ' + str(data_size[j]))
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
            if (tp+fn) > 0:
                tpr_test.append(tp/(tp+fn))
            else:
                tpr_test.append(0)
            if (tn+fp) > 0:
                fpr_test.append(tn/(tn+fp))
            else:
                fpr_test.append(0)
            #tpr_test.append(tpr[0])
            #fpr_test.append(fpr[0])
            acc_test.append(classifier.score(X_test, y_test))


            # Classify using validation data set
            classifier = KNeighborsClassifier(n_neighbors=int(knn_global_best["position"][0]))
            classifier.fit(X_input, y_input)
            validate_probs = classifier.predict_proba(X_validate)
            inpsize_val.append(data_size[j])
            # Compute ROC curve and ROC area for each class of validation data
            y_validate_bin = np.empty((len(y_validate), len(label_map)))
            for k in range(y_validate_bin.shape[0]):
                arr = [0 for _ in range(len(label_map))]
                arr[labels[k]] = 1
                y_validate_bin[k] = np.asarray(arr)
            print('Test Label original shape: ' + str(np.asarray(y_validate).shape))
            print('Test Label binary shape: ' + str(y_validate_bin.shape))
            print('Test score shape:' + str(validate_probs.shape))
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for k in range(len(label_map)):
                if k == als_patient_label:
                    fpr[0], tpr[0], _ = roc_curve(y_validate_bin[:, k], validate_probs[:, k])
                    roc_auc[0] = auc(fpr[0], tpr[0])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_validate_bin.ravel(), validate_probs.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            plt.subplot(2, 1, 2)
            plt.title("Validation data")
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.plot(fpr[0], tpr[0], lw=lw,
                     label='ROC curve (area = %0.2f)' % roc_auc[0] + ', Input Size: ' + str(data_size[j]))

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
                    if y_validate[k] == np.argmax(validate_probs[k]):
                        tp += 1
                    else:
                        fn += 1
                else:
                    if y_validate[k] == np.argmax(validate_probs[k]):
                        tn += 1
                    else:
                        fp += 1
            if (tp+fn) > 0:
                tpr_val.append(tp/(tp+fn))
            else:
                tpr_val.append(0)
            if (tn+fp) > 0:
                fpr_val.append(tn/(tn+fp))
            else:
                fpr_val.append(0)
            #tpr_val.append(tpr[0][-1])
            #fpr_val.append(fpr[0][-1])
            acc_val.append(classifier.score(X_validate, y_validate))
        # Plot Validation Accuracy, Test Accuracy, Validation Specificity and Test Sensitivity
        plt.figure(fig_num+10+i)
        plt.xlabel('No. of Input Data')
        plt.ylabel('Performance')
        plt.suptitle('Performance Graph - Feature: ' + str(classification_feature_labels[i].upper()))
        plt.plot(inpsize_test, np.asarray(acc_test)*100, label='Test Accuracy')
        plt.plot(inpsize_test, np.asarray(acc_val)*100, label='Validation Accuracy')
        plt.plot(inpsize_test, np.asarray(tpr_test)*100, label='Sensitivity')
        plt.plot(inpsize_test, np.asarray(fpr_test)*100, label='Specificity')
        plt.grid()
        plt.legend(loc='upper left')

        #classification_result_table.field_names = ["SL No.", "Feature", "Avg. Test Acc.", "Avg. Test Specificity",
        #                               "Avg. Test Sensitivity"]
        sl_no = len(classification_result_table_test._rows) + 1
        feature = classification_feature_labels[i].upper()
        avg_test_acc = np.average(np.asarray(acc_test)*100)
        avg_test_sensitivity = np.average(np.asarray(tpr_test)*100)
        avg_test_specificity = np.average((1-np.asarray(fpr_test))*100)
        classification_result_table_test.add_row([sl_no, feature, avg_test_acc, avg_test_specificity, avg_test_sensitivity])

        sl_no = len(classification_result_table_val._rows) + 1
        avg_val_acc = np.average(np.asarray(acc_val) * 100)
        avg_val_sensitivity = np.average(np.asarray(tpr_val) * 100)
        avg_val_specificity = np.average(np.asarray(fpr_val) * 100)
        classification_result_table_val.add_row([sl_no, feature, avg_val_acc, avg_val_specificity, avg_val_sensitivity])

    print("Test Performance: ")
    print(classification_result_table_test.get_string())
    with open(classification_result_path_test, 'w') as f:
        f.write(classification_result_table_test.get_html_string())

    print("Validation performance: ")
    print(classification_result_table_val.get_string())
    with open(classification_result_path_val, 'w') as f:
        f.write(classification_result_table_val.get_html_string())

    plt.show()








