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
import MyNet.emg_classification_library.muap_analysis_functions as mfunctions


# Expands a MUAP waveform from the Signal Waveform
def expand_muap_waveform(data, fs, muaps, muap_firing_index, expand_duration, verbose=True):

    signal_expand_length = int(math.ceil((fs * expand_duration) / 1000))
    expanded_signals = []
    for j in range(len(muaps)):
        if verbose:
            print('Expanding MUAP Waveform No. ' + str(j) + ", Left: " + str(len(muaps) - j))
        expand_crop_start = muap_firing_index[j] - int(signal_expand_length / 2)
        expand_crop_end = muap_firing_index[j] + int(signal_expand_length / 2) + 1
        expanded_signal = filtered[expand_crop_start:expand_crop_end]
        if verbose:
            print('Original MUAP Length: ' + str(len(muaps[i])))
            print('Expanded Signal length: ' + str(len(expanded_signal)))
        expanded_signals.append(expanded_signal)
    return expanded_signals
# Extracts features from MUAP Decomposition output of each signal for classification according to the paper
def default_feature_extraction(data, sampling_rates, waveforms, classes, firing_time, firing_index, firing_table, residue,
                               expand_duration=25, verbose=True, plot=False):
    verbose = True
    """
    :param data: The list of filtered signal in the data set
    :param sampling_rates: Sampling rates of the filtered signals of the data set
    :param waveforms: The list of MUAP waveforms of the filtered signals in the data set
    :param classes: The list of MUAP classes of the filtered signals in the data set
    :param firing_time: The list of firing times og the MUAP waveforms of the filtered signals in the data set
    :param firing_index: The indices of filtered data where each MUAP waveform of the filtered data fires
    :param firing_table: The firing table for all output neurons
    :param residue: The residue waveforms of the decomposed MUAP
    :param verbose: Whether progress should be displayed
    :return: Features extracted from each filtered waveform
    """
    # Parameter Measurement
    extracted_features = []
    # For each filtered data
    for i in range(len(data_filtered)):
        if verbose:
            print('Averaging actual MUAP Waveforms for extracting features')
            print('Total MUAPs: ' + str(len(waveforms[i])))
        # Get the current filtered signal, MUAP waveforms, firing indices, firing times and sampling rate
        filtered = data_filtered[i]
        fs = sampling_rates[i]
        muaps = waveforms[i]
        muap_classes = classes[i]
        muap_firing_times = firing_time[i]
        muap_firing_index = firing_index[i]

        # Expand the MUAP waveform duration for averaging and extracting features
        expanded_signals = expand_muap_waveform(filtered, fs, muaps, muap_firing_index, expand_duration, verbose=verbose)

        if len(expanded_signals) > 0:
            # For each output neuron/node in the firing table
            if verbose:
                print("Caculating Average and Standard Deviation for each sample point of all MUAP waveforms in a class")
            for j in range(len(firing_table)):
                # Calculate Average and Standard Deviation for each MUAP waveform belonging to that output class
                avg_sample_points = np.asarray([0 for _ in range(len(expanded_signals[0]))], dtype=np.float64)
                points = [ [] for _ in range(len(expanded_signals[0]))]
                if verbose:
                    print("Calculating Average MUAP waveform for output node No. " + str(j+1))
                    print("Calculating Average and Std for " + str(len(expanded_signals[0])) + " sample points")
                for k in range(len(expanded_signals)):
                    if verbose:
                        print("Calculating Average and Std for MUAP waveform No. " + str(k+1))
                    if muap_classes[k] == j:
                        avg_sample_points += np.asarray(expanded_signals[k])
                        if verbose:
                            print("Current Average Waveform for the output: " + str(avg_sample_points.shape))
                        for m in range(len(expanded_signals[k])):
                            points[m].append(expanded_signals[k][m])
                std_dev = [np.std(points[k]) for k in range(len(points))]
                avg_sample_points = avg_sample_points / len(expanded_signals)
                average_signal = []
                for k in range(len(points)):
                    if verbose:
                        print("Averaging sample point No. " + str(k))
                        print("Average data in the sample point: " + str(avg_sample_points[k]))
                        print("Standard Deviation of data in the sample point: " + str(std_dev[k]))
                    if -1 <  < 1:
                        average_signal.append(avg_sample_points[k])
                # Calculate Average for the class whose
                if verbose:
                    print("MUAP length before averaging: " + str(len(expanded_signals[0])))
                    print("Average MUAP length: " + str(len(average_signal)))
    return extracted_features

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
    

    """
    verbose = True
    # ----------------- Data Acquisition Parameters--------------------------------
    if verbose:
        print("SETTING DATA ACQUISITION PARAMETERS...")
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
    result_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\emg_classification_library\\pso_svm_classification_output\\'

    print("LOADING DATA SET")
    raw_urls, raw_labels, label_map = dfunctions.get_dataset(data_base_dir, shuffle=True)
    raw_urls = list(raw_urls)[:2]
    raw_labels = list(raw_labels)[:2]
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

    # --------------------------Data Preprocessing Parameters--------------------------------------
    # Cropping
    cropped_signal_duration = 5000  # Cropped Signal length(ms)

    # Filtering
    signal_filter_band = 'lowpass'  # Signal Filter type
    signal_filter_range = [8000]  # Lowpass: 8KHz
    signal_filter_order = 2  # Filter Order

    # ------------------------- MUAP Segmentation Parameters---------------------------------------
    # Segmentation/MUAP waveform Calculation
    muap_waveform_duration = 6  # Duration of each identified MUAP waveform segment(ms)

    # The minimum amount by which a peak's amplitude must increase in order to be counted as valid MUAP(uV)
    muap_peak_rise_threshold = 40
    # The duration within which the amplitude of a peak must exceed the specified threshold
    muap_rise_duration = 0.1

    # ------------------------- MUAP Classification Parameters--------------------------------------
    # The number of output neurons in the SOFM network
    muap_output_neurons = 8

    # ------------------------------1. DATA ACQUISITION & PREPROCESSING-------------------------------------------------
    """
    This section loads raw signal data from the urls and arranges it in an array for preprocessing.
    The steps followed are:
    1. For each URL:
        1.1 Load Numpy data.
        1.2 Read Sampling rate
        1.3 Pad/Crop raw Input data in order to make all sample data of same length.
        1.4 Store the Cropped data and their corresponding labels
        1.5 Filter the Cropped data and store the filtered data
    """
    if verbose:
        print('ACQUIRING & PREPROCESSING SIGNALS FROM DATA SET...')
    data_np = []
    data_cropped = []
    data_filtered = []
    data_labels = []
    data_fs = []
    data_acq_verbose = False
    data_acq_plot = 'none'  # interactive, debug or none
    data_acq_plot_fig_num = 100

    if data_acq_plot == 'interactive' or data_acq_plot == 'debug':
        plt.figure(data_acq_plot_fig_num)
    if data_acq_plot == 'interactive':
        plt.ion()
        plt.show()
    for i in range(len(urls)):
        if verbose:
            print("Loading data No. " + str(i) + ", Left: " + str(len(urls)-i))
        # Load Numpy data
        d = np.load(os.path.join(urls[i], data_filename))
        data_np.append(d)  # Add raw data to list
        data_labels.append(labels[i])  # Add class label to list
        # Read Sampling rate
        fs = dfunctions.read_sampling_rate(os.path.join(urls[i], header_filename))
        data_fs.append(fs)  # Add sampling rate to list

        # Crop data
        cropped_data_length = int( (fs*cropped_signal_duration)/1000 )  # Number of samples to be kept after cropping
        extra = len(d) - cropped_data_length  # The number of samples to be cropped

        # If the loaded data length is greater than required cropped data length
        if extra > 0:
            crop_left = int(extra/2)  # Crop left end of the signal
            crop_right = extra - crop_left  # Crop right end of the signal
            cropped_data = d[crop_left:-crop_right]  # Crop the signal
        # Else if the data length is less than the required cropped data length
        elif extra < 0:
            zeros_left = int(abs(extra)/2)  # Pad left end of the signal with zeroes
            zeroes_right = abs(extra) - zeros_left  # Pad right end of the signal with zeroes
            # Create zero padded cropped data
            cropped_data = np.asarray(
                [0 for _ in range(zeros_left)] + d.tolist() + [0 for _ in range(zeroes_right)]
            )
        # Else if the length of data is equal to the required length of cropped data
        else:
            cropped_data = d.copy()  # Add the data as cropped data


        data_cropped.append(cropped_data)  # Store Cropped data

        # Filter the cropped data
        filtered = sfunctions.butter_bandpass_filter(cropped_data, signal_filter_range, signal_filter_band,
                                                     fs, order=signal_filter_order)

        data_filtered.append(filtered)  # Add the filtered data to list of filtered data
        if data_acq_verbose:
            print("Loaded data from: " + urls[i])
            print("Subject Type: " + str(label_map[labels[i]]))
            print("Original Signal duration: " + str( (1000*d.shape[0])/fs ) + "ms")
            print("Cropped Signal duration: " + str((1000 * cropped_data.shape[0]) / fs) + "ms")
            print("Filtered data duration: " + str( (1000 * filtered.shape[0]) / fs) + "ms")
            print("----------------------------------------------------------------------------------\n\n\n")

        if data_acq_plot == 'interactive' or data_acq_plot == 'debug':
            plt.clf()
            plt.suptitle("EMG Preprocessing: Patient type-" + label_map[labels[i]].upper())
            plt.subplot(3, 1, 1)
            plt.title("Raw EMG Signal - Duration: " + str(int(len(d)/fs)) + " seconds")
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
            plt.grid()
            plt.subplots_adjust(hspace=0.7)
            plt.plot(d)

            plt.subplot(3, 1, 2)
            plt.title("Cropped EMG Signal - Duration: " + str(int(len(cropped_data) / fs)) + " seconds")
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
            plt.grid()
            plt.subplots_adjust(hspace=0.7)
            plt.plot(cropped_data)

            plt.subplot(3, 1, 3)
            plt.title("Filtered EMG Signal - Duration: " + str(int(len(filtered) / fs)) + " seconds")
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
            plt.grid()
            plt.subplots_adjust(hspace=0.7)
            plt.plot(filtered)
            if data_acq_plot == 'interactive':
                plt.pause(2)
            else:
                plt.show()

    # ------------------------------2. MUAP SEGMENTATION, CLASSIFICATION & DECOMPOSITION-------------------------------------------------
    """
    This section detects MUAP waveforms & their corresponding firing time using Peak Detection and
    window technique. The steps followed are:
    1. For each filtered data:
        1.1 Calculate Peaks of from the signal using thresholding technique.
        1.2 Calculate & Store Candidate MUAP waveforms & their firing time from the detected peaks and the corresponding
            signal.
        1.3 Calculate & Store final MUAP Waveforms(Actual & Superimposed), Classes and the firing table from Candidate
            MUAP Waveforms.        
        1.4 Decompose the final MUAP superimposed waveforms based on actual waveforms and update the firing table
            accordingly.    

    """
    print('PERFORMING MUAP SEGMENTATION, CLASSIFICATION AND DECOMPOSITION FOR SIGNALS OF DATA SET...')
    all_candidate_muap_waveforms = []  # Stores the candidate identified MUAP waveform
    all_candidate_muap_firing_times = []  # Stores the candidate firing times of identified waveforms
    all_candidate_waveform_peak_indices = []  # Stores the index of the peak of MUAP waveform in the filtered signal
    all_final_muap_waveforms = [] # Stores the final indentified MUAP waveforms
    all_final_muap_output_classes = [] # Stores the final output classes of identified waveforms
    all_final_muap_superimposition_classes = []  # Stores final superimposition classes of identified waveforms
    all_final_muap_firing_times = []  # Stores final firing time for identified MUAP Waveforms
    all_final_muap_firing_tables = []  # Stores firing table of all signals
    all_decomposed_actual_waveforms = []
    all_decomposed_actual_waveform_classes = []
    all_decomposed_actual_waveform_firing_time = []
    all_decomposed_actual_waveform_indices = []
    all_decomposed_residue_waveforms = []
    all_decomposed_muap_firing_table = []

    muap_seg_verbose = False
    muap_seg_plot = 'none'  # debug, interactive or none
    muap_seg_plot_fig_num = 200


    if muap_seg_plot == 'debug' or muap_seg_plot == 'interactive':
        plt.figure(muap_seg_plot_fig_num)
        plt.figure(muap_seg_plot_fig_num+1)
        plt.figure(muap_seg_plot_fig_num + 2)

    if muap_seg_plot == 'interactive':
        plt.ion()
        plt.show()
    # For each signal in the data set
    for i in range(len(data_filtered)):
        if verbose:
            print('Loading data No. ' + str(i) + ", Left: " + str(len(data_filtered)-i))
        if muap_seg_verbose:
            print("Calculating MUAP waveforms from " + urls[i] + " : Left - " + str(len(filtered)-i))

        filtered = data_filtered[i]
        muap_waveform_samples = int(math.ceil((fs * muap_waveform_duration) / 1000))
        # Calculate the candidate MUAP waveforms, their firing time and Peak indices

        #potential_muap, muap_waveforms, muap_firing_times = muap_an_get_segmentation_const_window(
        # data, fs, window_ms=6, mav_coeff=30, peak_amp_increase_amount=40, peak_amp_increase_duration=0.1,
        # verbose=True)

        peak_indices, waveforms, firing_time = mfunctions.muap_an_get_segmentation_const_window(
            filtered.copy(), data_fs[i], window_ms=muap_waveform_duration,
            peak_amp_increase_amount=muap_peak_rise_threshold, peak_amp_increase_duration=muap_rise_duration,
            verbose=muap_seg_verbose, window_length=muap_waveform_samples
        )
        all_candidate_waveform_peak_indices.append(peak_indices)
        all_candidate_muap_waveforms.append(waveforms)
        all_candidate_muap_firing_times.append(firing_time)

        # Classify Candidate MUAP waveforms to Motor Unit classes and detect superimposed waveforms

        # The size of the SOFM Network
        network_size = [muap_output_neurons, muap_waveform_samples]
        # Firing Table for the Classified MUAPs
        muap_firing_table = [[] for _ in range(muap_output_neurons)]

        """
        custom_muap_sofm_classification(muaps, muap_firing_time, muap_firing_table, muap_size=[8, 120], lvq2=False,
                                        init_weight=0.0001, init_mid_weight_const=0.1, g=1, lvq_gaussian_hk=0.2,
                                        epochs=1,
                                        lvq_gaussian_thresh=0.005, learning_rate=1, neighborhood='gaussian',
                                        verbose=True)
        Returns: MUAP Waveforms(List[[]]), MUAP Class(List[]), Classification Output(Actual/Superimposed)(List[]),
                 MUAP Firing Times(List[[]]), Firing Table(List[[]])                                
        """

        final_muap_waveforms, final_muap_classes, final_muap_outputs, final_muap_firing_time, muap_firing_table = mfunctions.custom_muap_sofm_classification(
            waveforms, firing_time, muap_firing_table, muap_size=network_size, verbose=muap_seg_verbose
        )
        all_final_muap_waveforms.append(final_muap_waveforms)
        all_final_muap_output_classes.append(final_muap_classes)
        all_final_muap_superimposition_classes.append(final_muap_outputs)
        all_final_muap_firing_times.append(final_muap_firing_time)
        all_final_muap_firing_tables.append(muap_firing_table)

        # Decompose superimposed MUAPs based on the Actual MUAPs and update the firing table
        decomposed_firing_table = [ [] for _ in range(muap_output_neurons)]


        #perform_emg_decomposition(waveforms, waveform_classes, waveform_superimposition, firing_time, max_residue_amp=30,
        #                threshold_const_a=0.5, threshold_const_b=4, nd_thresh1=0.2, nd_thresh2=0.5,
         #               calculate_endpoints=False, pearson_correlate=True, plot=False, verbose=True)
        # @Returns:actual_muaps(List[ [waveform[], classes[], firing time[]] ]), residue_superimposed_muaps (List[ [waveform[], classes[], firing time[]] ])

        decomposed_actual_muaps, decomposed_residue_muap = mfunctions.perform_emg_decomposition(
            final_muap_waveforms, final_muap_classes, final_muap_outputs, final_muap_firing_time, verbose=muap_seg_verbose
        )
        decomposed_actual_muap_waveforms = [decomposed_actual_muaps[j][0] for j in range(len(decomposed_actual_muaps))]
        decomposed_actual_muap_classes = [decomposed_actual_muaps[j][1] for j in range(len(decomposed_actual_muaps))]
        decomposed_actual_muap_firing_time = [decomposed_actual_muaps[j][2] for j in range(len(decomposed_actual_muaps))]
        decomposed_actual_muap_index = [decomposed_actual_muaps[j][3] for j in range(len(decomposed_actual_muaps))]
        residue_actual_muap_waveforms = [decomposed_residue_muap[j][0] for j in range(len(decomposed_residue_muap))]
        residue_actual_muap_classes = [decomposed_residue_muap[j][1] for j in range(len(decomposed_residue_muap))]
        residue_actual_muap_firing_time = [decomposed_residue_muap[j][2] for j in range(len(decomposed_residue_muap))]
        residue_actual_muap_index = [decomposed_residue_muap[j][3] for j in range(len(decomposed_residue_muap))]
        for j in range(len(decomposed_actual_muap_firing_time)):
            for k in range(len(decomposed_firing_table)):
                if decomposed_actual_muap_classes[j] == k:
                    decomposed_firing_table[k] += decomposed_actual_muap_firing_time[j]
                    break

        all_decomposed_actual_waveforms.append(decomposed_actual_muap_waveforms)
        all_decomposed_actual_waveform_classes.append(decomposed_actual_muap_classes)
        all_decomposed_actual_waveform_firing_time.append(decomposed_actual_muap_firing_time)
        all_decomposed_residue_waveforms.append(residue_actual_muap_waveforms)
        all_decomposed_muap_firing_table.append(decomposed_firing_table)
        all_decomposed_actual_waveform_indices.append(decomposed_actual_muap_index)

        if muap_seg_plot == 'debug' or muap_seg_plot == 'interactive':
            colors = ['green', 'red', 'lightcoral', 'black', 'magenta', 'orange', 'dimgrey', 'lightseagreen']
            plt.figure(muap_seg_plot_fig_num)
            plt.clf()
            plt.suptitle("MUAP Segmentation: Patient type-" + label_map[labels[i]].upper())
            plt.subplot(3, 1, 1)
            plt.title('Candidate MUAP Waveforms - Total: ' + str(len(waveforms)))
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
            plt.grid()
            plt.subplots_adjust(hspace=0.7)
            plt.plot(np.arange(0, len(filtered), 1), filtered, 'b-', linewidth=0.5)
            for j in range(len(waveforms)):
                crop_start = peak_indices[j] - int(muap_waveform_samples/2)
                crop_end = crop_start + int(muap_waveform_samples)
                plt.plot(np.arange(crop_start, crop_end, 1), waveforms[j], 'r-')
                plt.plot(peak_indices[j], filtered[peak_indices[j]], 'gx')

            plt.subplot(3, 1, 2)
            plt.title("MUAP Firing Table - Total Motor Units: " + str(muap_output_neurons))
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
            plt.grid()
            plt.subplots_adjust(hspace=0.7)
            plt.plot(np.arange(0, len(filtered), 1), filtered, 'b-', linewidth=0.5)
            for j in range(len(muap_firing_table)):
                for k in range(len(muap_firing_table[j])):
                    peak = muap_firing_table[j][k]
                    crop_start = peak - int(muap_waveform_samples/2)
                    crop_end = crop_start + int(muap_waveform_samples)
                    plt.plot(np.arange(crop_start, crop_end, 1),
                             filtered[np.arange(crop_start, crop_end, 1)], color=colors[j])
                    plt.plot(peak, filtered[peak], marker='x', color=colors[j])

            plt.subplot(3, 1, 3)
            plt.title("MUAP Decomposition Firing Table")
            plt.xlabel("Samples[n]")
            plt.ylabel("Amplitude(uV)")
            plt.grid()
            plt.subplots_adjust(hspace=0.7)
            plt.plot(np.arange(0, len(filtered), 1), filtered, 'b-', linewidth=0.5)
            for j in range(len(decomposed_firing_table)):
                for k in range(len(decomposed_firing_table[j])):
                    peak = decomposed_firing_table[j][k]
                    crop_start = peak - int(muap_waveform_samples/2)
                    crop_end = crop_start + int(muap_waveform_samples)
                    plt.plot(np.arange(crop_start, crop_end, 1),
                             filtered[np.arange(crop_start, crop_end, 1)], color=colors[j])
                    if peak in decomposed_actual_muap_index:
                        plt.plot(peak, filtered[peak], marker='x', color='blue')
                    else:
                        plt.plot(peak, filtered[peak], marker='x', color=colors[j])

            plt.figure(muap_seg_plot_fig_num+1)
            plt.clf()
            plt.suptitle("MUAP Classification: : Patient type-" + label_map[labels[i]].upper())
            cols = 2
            rows = int(math.ceil(muap_output_neurons/cols))
            for j in range(muap_output_neurons):
                plt.subplot(rows, cols, j+1)

                plt.title("MU" + str(j+1) + ": Total: " + str(list(final_muap_classes).count(j)))
                plt.xlabel("Samples[n]")
                plt.ylabel("Amplitude(uV)")
                plt.grid()
                plt.subplots_adjust(hspace=0.9, wspace=0.3)
                for k in range(len(final_muap_waveforms)):
                    if final_muap_classes[k] == j: # Actual & Superimposed Waveform display
                        plt.plot(final_muap_waveforms[k], color=colors[j])

            plt.figure(muap_seg_plot_fig_num + 2)
            plt.clf()
            plt.suptitle("MUAP Decomposition: : Patient type-" + label_map[labels[i]].upper())
            cols = 2
            rows = int(math.ceil(muap_output_neurons / cols))
            for j in range(muap_output_neurons):
                plt.subplot(rows, cols, j + 1)
                total = 0
                for k in range(len(decomposed_actual_muap_classes)):
                    if decomposed_actual_muap_classes[k] == j:
                        total += 1
                plt.title("MU" + str(j + 1) + ": Total: " + str(total))
                plt.xlabel("Samples[n]")
                plt.ylabel("Amplitude(uV)")
                plt.grid()
                plt.subplots_adjust(hspace=0.9, wspace=0.3)
                for k in range(len(decomposed_actual_muap_classes)):
                    if decomposed_actual_muap_classes[k] == j:  # Actual Waveform display
                        plt.plot(decomposed_actual_muap_waveforms[k], color=colors[j])

            if muap_seg_plot == 'interactive':
                plt.pause(2)
            else:
                plt.show()


    # ------------------------------4. FEATURE EXTRACTION-------------------------------------------------
    """
    This section detects extracts features from the MUAP Waveform and firing table of each output motor unit of each
    filtered data. The steps followed are:
    1. For each decomposed data:
        1.1  Extract feature from the MUAP waveforms and Firing Table of the decomposed data       

    """
    if verbose:
        print('EXTRACTING FEATURES FROM SIGNALS OF THE DATA SET...')
    # default_feature_extraction(data,sampling_rates, waveforms, classes, firing_time, firing_index, firing_table, residue)
    data_features = default_feature_extraction(data_filtered, data_fs,
        all_decomposed_actual_waveforms, all_decomposed_actual_waveform_classes, all_decomposed_actual_waveform_firing_time,
        all_decomposed_actual_waveform_indices, all_decomposed_muap_firing_table, all_decomposed_residue_waveforms
    )