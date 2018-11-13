import os, random,sys, pdb, datetime, collections, math
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable, from_html_one
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.model_selection import train_test_split

import MyNet.emg_classification_library.dataset_functions as dfunctions
import MyNet.emg_classification_library.signal_analysis_functions as sfunctions
import MyNet.emg_classification_library.muap_analysis_functions as mfunctions
import MyNet.emg_classification_library.feature_extraction_functions as ffunctions
import MyNet.emg_classification_library.classifier_functions as cfunctions

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
    result_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\emg_classification_library\\emg_decomposition_classification_output\\'

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
    data_size = [len(urls)]
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

    #--------------------------Classification Parameters---------------------------------------------
    classifier_names = ["SVM(RBF)", "SVM(POLY)", "KNN"]
    classifier_objects = [cfunctions.getSVMRBF, cfunctions.getSVMPOL, cfunctions.getKNN]
    classification_result_path_test = os.path.join(result_base_dir, "average_performance_graph_test" + suffix + ".html")
    classification_result_path_val = os.path.join(result_base_dir,
                                                  "average_performance_graph_validation" + suffix + ".html")
    if os.path.exists(classification_result_path_test):
        with open(classification_result_path_test, 'r') as f:
            classification_result_table_test = from_html_one(f.read())
    else:
        classification_result_table_test = PrettyTable()
        classification_result_table_test.field_names = ["SL No.", "Classifier", "Avg. Test Acc.",
                                                        "Avg. Test Specificity",
                                                        "Avg. Test Sensitivity"]
    if os.path.exists(classification_result_path_val):
        with open(classification_result_path_val, 'r') as f:
            classification_result_table_val = from_html_one(f.read())
    else:
        classification_result_table_val = PrettyTable()
        classification_result_table_val.field_names = ["SL No.", "Classifier", "Avg. Validation Acc.",
                                                       "Avg. Validation Specificity",
                                                       "Avg. Validation Sensitivity"]

    #--------------------------RESULT FILES AND PARAMETERS--------------------------------------
    save_features = True
    feature_numpy_result_file = os.path.join(result_base_dir, 'features' + suffix + '.npy')
    feature_label_result_file = os.path.join(result_base_dir, 'labels' + suffix + '.npy')
    feature_label_map_result_file = os.path.join(result_base_dir, 'label_map' + suffix + '.npy')


    if not os.path.exists(feature_numpy_result_file):
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
                print("Loading data No. " + str(i) + ", Left: " + str(len(urls) - i))
            # Load Numpy data
            d = np.load(os.path.join(urls[i], data_filename))
            data_np.append(d)  # Add raw data to list
            data_labels.append(labels[i])  # Add class label to list
            # Read Sampling rate
            fs = dfunctions.read_sampling_rate(os.path.join(urls[i], header_filename))
            data_fs.append(fs)  # Add sampling rate to list

            # Crop data
            cropped_data_length = int(
                (fs * cropped_signal_duration) / 1000)  # Number of samples to be kept after cropping
            extra = len(d) - cropped_data_length  # The number of samples to be cropped

            # If the loaded data length is greater than required cropped data length
            if extra > 0:
                crop_left = int(extra / 2)  # Crop left end of the signal
                crop_right = extra - crop_left  # Crop right end of the signal
                cropped_data = d[crop_left:-crop_right]  # Crop the signal
            # Else if the data length is less than the required cropped data length
            elif extra < 0:
                zeros_left = int(abs(extra) / 2)  # Pad left end of the signal with zeroes
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
                print("Original Signal duration: " + str((1000 * d.shape[0]) / fs) + "ms")
                print("Cropped Signal duration: " + str((1000 * cropped_data.shape[0]) / fs) + "ms")
                print("Filtered data duration: " + str((1000 * filtered.shape[0]) / fs) + "ms")
                print("----------------------------------------------------------------------------------\n\n\n")

            if data_acq_plot == 'interactive' or data_acq_plot == 'debug':
                plt.clf()
                plt.suptitle("EMG Preprocessing: Patient type-" + label_map[labels[i]].upper())
                plt.subplot(3, 1, 1)
                plt.title("Raw EMG Signal - Duration: " + str(int(len(d) / fs)) + " seconds")
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
        all_final_muap_waveforms = []  # Stores the final indentified MUAP waveforms
        all_final_muap_output_classes = []  # Stores the final output classes of identified waveforms
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
            plt.figure(muap_seg_plot_fig_num + 1)
            plt.figure(muap_seg_plot_fig_num + 2)

        if muap_seg_plot == 'interactive':
            plt.ion()
            plt.show()
        # For each signal in the data set
        for i in range(len(data_filtered)):
            if verbose:
                print('Loading data No. ' + str(i) + ", Left: " + str(len(data_filtered) - i))
            if muap_seg_verbose:
                print("Calculating MUAP waveforms from " + urls[i] + " : Left - " + str(len(filtered) - i))

            filtered = data_filtered[i]
            muap_waveform_samples = int(math.ceil((fs * muap_waveform_duration) / 1000))
            # Calculate the candidate MUAP waveforms, their firing time and Peak indices

            # potential_muap, muap_waveforms, muap_firing_times = muap_an_get_segmentation_const_window(
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
            decomposed_firing_table = [[] for _ in range(muap_output_neurons)]

            # perform_emg_decomposition(waveforms, waveform_classes, waveform_superimposition, firing_time, max_residue_amp=30,
            #                threshold_const_a=0.5, threshold_const_b=4, nd_thresh1=0.2, nd_thresh2=0.5,
            #               calculate_endpoints=False, pearson_correlate=True, plot=False, verbose=True)
            # @Returns:actual_muaps(List[ [waveform[], classes[], firing time[]] ]), residue_superimposed_muaps (List[ [waveform[], classes[], firing time[]] ])

            decomposed_actual_muaps, decomposed_residue_muap = mfunctions.perform_emg_decomposition(
                final_muap_waveforms, final_muap_classes, final_muap_outputs, final_muap_firing_time,
                verbose=muap_seg_verbose
            )
            decomposed_actual_muap_waveforms = [decomposed_actual_muaps[j][0] for j in
                                                range(len(decomposed_actual_muaps))]
            decomposed_actual_muap_classes = [decomposed_actual_muaps[j][1] for j in
                                              range(len(decomposed_actual_muaps))]
            decomposed_actual_muap_firing_time = [decomposed_actual_muaps[j][2] for j in
                                                  range(len(decomposed_actual_muaps))]
            decomposed_actual_muap_index = [decomposed_actual_muaps[j][3] for j in range(len(decomposed_actual_muaps))]
            residue_actual_muap_waveforms = [decomposed_residue_muap[j][0] for j in range(len(decomposed_residue_muap))]
            residue_actual_muap_classes = [decomposed_residue_muap[j][1] for j in range(len(decomposed_residue_muap))]
            residue_actual_muap_firing_time = [decomposed_residue_muap[j][2] for j in
                                               range(len(decomposed_residue_muap))]
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
                    crop_start = peak_indices[j] - int(muap_waveform_samples / 2)
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
                        crop_start = peak - int(muap_waveform_samples / 2)
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
                        crop_start = peak - int(muap_waveform_samples / 2)
                        crop_end = crop_start + int(muap_waveform_samples)
                        plt.plot(np.arange(crop_start, crop_end, 1),
                                 filtered[np.arange(crop_start, crop_end, 1)], color=colors[j])
                        if peak in decomposed_actual_muap_index:
                            plt.plot(peak, filtered[peak], marker='x', color='blue')
                        else:
                            plt.plot(peak, filtered[peak], marker='x', color=colors[j])

                plt.figure(muap_seg_plot_fig_num + 1)
                plt.clf()
                plt.suptitle("MUAP Classification: : Patient type-" + label_map[labels[i]].upper())
                cols = 2
                rows = int(math.ceil(muap_output_neurons / cols))
                for j in range(muap_output_neurons):
                    plt.subplot(rows, cols, j + 1)

                    plt.title("MU" + str(j + 1) + ": Total: " + str(list(final_muap_classes).count(j)))
                    plt.xlabel("Samples[n]")
                    plt.ylabel("Amplitude(uV)")
                    plt.grid()
                    plt.subplots_adjust(hspace=0.9, wspace=0.3)
                    for k in range(len(final_muap_waveforms)):
                        if final_muap_classes[k] == j:  # Actual & Superimposed Waveform display
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
        feature_extract_verbose = True
        feature_extract_plot = 'none'  # debug, interactive or none
        feature_extract_plot_fig_num = 300
        fig_num = -1
        if verbose:
            print('EXTRACTING FEATURES FROM SIGNALS OF THE DATA SET...')
        if feature_extract_plot == 'debug' or feature_extract_plot == 'interactive':
            fig_num = feature_extract_plot_fig_num
        # default_feature_extraction(data,sampling_rates, waveforms, classes, firing_time, firing_index, firing_table, residue)
        data_features, feature_labels = ffunctions.default_feature_extraction(data_filtered.copy(), data_labels,
                                                                              label_map, data_fs,
                                                                              all_decomposed_actual_waveforms,
                                                                              all_decomposed_actual_waveform_classes,
                                                                              all_decomposed_actual_waveform_firing_time,
                                                                              all_decomposed_actual_waveform_indices,
                                                                              all_decomposed_muap_firing_table,
                                                                              all_decomposed_residue_waveforms,
                                                                              verbose=feature_extract_verbose,
                                                                              plot_fig_num=fig_num,
                                                                              plot_mode=feature_extract_plot
                                                                              )

        print("Total Features: " + str(np.asarray(data_features).shape))
        print('Total labels: ' + str(len(feature_labels)))

        if save_features:
            np.save(feature_numpy_result_file, data_features, allow_pickle=True)
            np.save(feature_label_result_file, np.asarray(feature_labels), allow_pickle=True)
            np.save(feature_label_map_result_file, np.asarray(label_map), allow_pickle=True)
    else:
        data_features = np.load(feature_numpy_result_file, allow_pickle=True)
        data_labels = list(np.load(feature_label_result_file, allow_pickle=True))
        label_map = list(np.load(feature_label_map_result_file, allow_pickle=True))

    # ------------------------------5. Classification-------------------------------------------------
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
    fig_num = 400
    lw = 2
    classification_verbose = True
    classification_plot = True
    # Create Input Vectors for the classifiers
    features = np.copy(data_features).reshape((len(data_features), -1))

    if scale_data:
        features = features / np.amax(np.abs(features))
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
            X = features[:data_size[j], :]  # Get Input feature
            y = labels[:data_size[j]]  # Get Input Label
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
                print("....Classification accuracy using test data: " + "{0:.2f}".format(
                    classifier.score(X_test, y_test) * 100))

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

