import os, random, sys, pdb, datetime, collections, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram, find_peaks
from sklearn.svm import SVC, SVR
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable, from_csv, from_html_one
from particle_swarm_optimization import Particle, PSO
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelBinarizer

import dataset_functions as dfunctions
import signal_analysis_functions as sfunctions
import muap_analysis_functions as mfunctions

# Expands a MUAP waveform from the Signal Waveform
def expand_muap_waveform(data, fs, muaps, muap_firing_index, expand_duration, verbose=True):

    signal_expand_length = int(math.ceil((fs * expand_duration) / 1000))
    expanded_signals = []
    for j in range(len(muaps)):
        if verbose:
            print('Expanding MUAP Waveform No. ' + str(j) + ", Left: " + str(len(muaps) - j))
        expand_crop_start = muap_firing_index[j] - int(signal_expand_length / 2)
        expand_crop_end = muap_firing_index[j] + int(signal_expand_length / 2) + 1
        expanded_signal = data[expand_crop_start:expand_crop_end]
        if verbose:
            print('Original MUAP Length: ' + str(len(muaps[j])))
            print('Expanded Signal length: ' + str(len(expanded_signal)))
        expanded_signals.append(expanded_signal)
    return expanded_signals

# Creates average class from a set of MUAP waveforms
def get_average_class(waveforms, verbose=True):
    if verbose:
        print("Calculating Average Class for " + str(len(waveforms[0])) + " points and " + str(len(waveforms)) + " MUAP waveforms")
    points_vals = [[] for _ in range(len(waveforms[0]))]
    for i in range(len(waveforms)):
        for j in range(len(waveforms[i])):
            points_vals[j].append(waveforms[i][j])
    points_avg = [np.average(points_vals[i]) for i in range(len(points_vals))]
    points_std = [np.std(points_vals[i]) for i in range(len(points_vals))]
    avg_class = [0 for _ in range(len(waveforms[0]))]
    avg_total = [0 for _ in range(len(waveforms[0]))]
    for i in range(len(points_vals)):
        if verbose:
            print('Calculating Average For point No. ' + str(i+1))
            print("Calculating Average Point Value from " + str(len(points_vals[i])) + " waveforms")
        for j in range(len(points_vals[i])):
            if points_avg[i]-np.abs(points_std[i]) <= points_vals[i][j] <= points_avg[i] + np.abs(points_std[i]):
                avg_class[i] += points_vals[i][j]
                avg_total[i] += 1
    result = []
    for i in range(len(avg_class)):
        if avg_total[i] > 0:
            result.append(avg_class[i] / avg_total[i])
        else:
            result.append(0)
    return result

# Corrects Baseline for a signal according to the procedure mentioned in paper
def get_baseline_corrected_waveform(data, baseline_correction_window_length, baseline_correction_window_height,
                                    verbose=True):
    mid_index = int(math.ceil(len(data)/2))
    avg_left = 0
    avg_right = 0

    # Calculate left Beginning Point
    for i in range(mid_index, -1, -baseline_correction_window_length):
        # From middle of the waveform, slide left using a window until the signal is enclosed in it
        if i-baseline_correction_window_length > 0:
            # If current position of the window is the last point where the window can be slided
            values_left_index = int(math.ceil(i-baseline_correction_window_length/2))  # Beginning point of the waveform
            if verbose:
                print('Beginning point: ' + str(values_left_index))
            # Get all values below the specified height of the window
            values_left = np.asarray(data[:values_left_index])
            #values_left = values_left[values_left <= baseline_correction_window_height]
            #values_left = values_left[values_left >= 0]
            if len(values_left) > 0:
                avg_left = np.average(values_left)
            else:
                avg_left = 0
    # Calculate Right Ending Point
    for i in range(mid_index, len(data), baseline_correction_window_length):
        # From middle of the waveform, slide right using a window until the signal is enclosed in it
        if i+baseline_correction_window_length < len(data):

            # If current position of the window is the last point where the window can be slided
            values_right_index = int(math.ceil(i+baseline_correction_window_length/2))  # Ending point of the waveform
            if verbose:
                print('End point: ' + str(values_right_index))
            # Get all values below the specified height of the window
            values_right = np.asarray(data[values_right_index:])
            #values_right = values_right[values_right <= baseline_correction_window_height]
            #values_right = values_right[values_right >= 0]
            if len(values_right) > 0:
                avg_right = np.average(values_right)
            else:
                avg_right = 0
    # Subtract Average left and right values from the waveform and return the baseline corrected waveform
    if verbose:
        print("Average left: " + str(avg_left))
        print("Average_right: " + str(avg_right))
    baseline_corrected = np.asarray(data) - avg_left
    baseline_corrected = baseline_corrected - avg_right
    return list(baseline_corrected)
# Extracts features from MUAP Decomposition output of each signal for classification according to the paper
def default_feature_extraction(data_filtered, labels, label_map, sampling_rates, waveforms, classes, firing_time,
                               firing_index, firing_table, residue, expand_duration=25, verbose=True, plot_fig_num=-1,
                               baseline_correction_window_duration=2, baseline_correction_window_amplitude=10,
                               plot_mode='none'):

    """
    :param data: The list of filtered signal in the data set
    :param data: The list of labels for the filtered signal
    :param label_map: The label map for the labels
    :param sampling_rates: Sampling rates of the filtered signals of the data set
    :param waveforms: The list of MUAP waveforms of the filtered signals in the data set
    :param classes: The list of MUAP classes of the filtered signals in the data set
    :param firing_time: The list of firing times og the MUAP waveforms of the filtered signals in the data set
    :param firing_index: The indices of filtered data where each MUAP waveform of the filtered data fires
    :param firing_table: The firing table for all output neurons
    :param residue: The residue waveforms of the decomposed MUAP
    :param verbose: Whether progress should be displayed
    :param plot_fig_num: Whether to plot the progress(< 0 is False & >0 indicates figure number)
    :param baseline_correction_window_duration: The length(ms) of window for baseline correction
    :param baseline_correction_window_amplitude: The height(uV) of window for baseline correction
    :return: Features extracted from each filtered waveform
    """
    # Parameter Measurement
    mu_expanded_signals = []
    mu_average_signals = []
    mu_baseline_corrected_signals = []
    extracted_features = []
    feature_labels = []
    if plot_mode == 'interactive':
        plt.ion()
        plt.show()
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
        muap_firing_table = firing_table[i]
        muap_expanded_signal = []
        muap_average_signal = []
        muap_baseline_corrected_signal = []
        muap_extracted_feature = []


        # Expand the MUAP waveform duration for averaging and extracting features
        expanded_signals = expand_muap_waveform(filtered, fs, muaps, muap_firing_index, expand_duration, verbose=verbose)

        if len(expanded_signals) > 0:
            # For each output neuron/node in the firing table
            if plot_fig_num > 0:
                cols = 2
                rows = int(math.ceil(len(muap_firing_table)/2))
                plt.figure(plot_fig_num)
                plt.clf()
                plt.suptitle("Actual MUAP Waveforms: " + str(len(muaps)) + " (" + label_map[labels[i]].upper() + ")")
                plt.figure(plot_fig_num+1)
                plt.clf()
                plt.suptitle("Expanded MUAP Waveforms" + " (" + label_map[labels[i]].upper() + ")")
                plt.figure(plot_fig_num+2)
                plt.clf()
                plt.suptitle("Average MUAP Waveforms" + " (" + label_map[labels[i]].upper() + ")")
                plt.figure(plot_fig_num+3)
                plt.clf()
                plt.suptitle("Baseline Corrected Average MUAP Waveforms" + " (" + label_map[labels[i]].upper() + ")")
            if verbose:
                print("Total Expanded Signals: " + str(len(expanded_signals)))
                print("Calculating Average and Standard Deviation for each sample point of all MUAP waveforms in a class")
            # For each Motor Unit
            for j in range(len(muap_firing_table)):
                # Get the expanded waveforms belonging to the MU
                expanded = []
                for k in range(len(expanded_signals)):
                    if muap_classes[k] == j and len(expanded_signals[k]) > 0:
                        expanded.append(expanded_signals[k])
                if verbose:
                    print("Total MUAP waveforms detected for Output node No. " + str(j+1) + " : " + str(len(expanded)))
                # Add the expanded signal list to the expanded signal list of all MUs
                muap_expanded_signal.append(expanded)
                # Caclulate average class from the expanded signals of the MU
                if len(expanded) > 0:
                    avg_class = get_average_class(expanded, verbose=verbose)
                else:
                    avg_class = []
                # Add the average class to the list of average classes for all MUss
                muap_average_signal.append(avg_class)

                # Calculate baseline correction for the average class
                if len(avg_class) > 0:
                    if verbose:
                        print("Correcting baseline for average class of length: " + str(len(avg_class)))
                    baseline_correction_window_length = int(math.ceil((fs*baseline_correction_window_duration)/1000))
                    baseline_correction_window_height = baseline_correction_window_amplitude
                    baseline_corrected = get_baseline_corrected_waveform(avg_class, baseline_correction_window_length,
                                                                         baseline_correction_window_height, verbose=verbose)
                else:
                    baseline_corrected = []
                # Add the baseline corrected waveforms of the average class to the list of baseline corrected wavforms
                # of all Mus
                muap_baseline_corrected_signal.append(baseline_corrected)

                # ------------------ Feature Extraction-------------------------
                if len(avg_class) > 0:
                    amplitude, amplitude_peaks = mfunctions.calculate_amplitude_difference([baseline_corrected.copy()])
                    amplitude = amplitude[0]
                    amplitude_peaks = amplitude_peaks[0]

                    duration, points = mfunctions.calculate_waveform_duration([baseline_corrected.copy()],
                                                                              (1000*len(baseline_corrected))/fs)
                    duration = duration[0]
                    points = points[0]

                    area = np.sum(np.abs(baseline_corrected[points[0]:points[1]]))
                    turns = mfunctions.calculate_turns([baseline_corrected.copy()], (1000*len(baseline_corrected))/fs, trim_muap=True)
                    turns = turns[0]
                    phases = mfunctions.calculate_phase([baseline_corrected.copy()], (1000*len(baseline_corrected))/fs)
                    phases = phases[0]

                    rise_time = (1000 * abs(amplitude_peaks[0]-amplitude_peaks[1]))/fs
                    muap_extracted_feature.append([amplitude, duration, area, turns, phases])


                else:
                    amplitude = -sys.maxsize
                    amplitude_peaks = [-1, -1]
                    duration = -sys.maxsize
                    points = [-1, -1]
                    area = -sys.maxsize
                    turns = -sys.maxsize
                    phases = -sys.maxsize
                    rise_time = -sys.maxsize
                    muap_extracted_feature.append([0, 0, 0, 0, 0])
                if verbose:
                    print("Average class shape: " + str(len(avg_class)))
                if plot_fig_num > 0:
                    plt.figure(plot_fig_num)
                    plt.subplot(rows, cols, j + 1)
                    plt.title("MU" + str(j + 1) + " : " + str(len(np.asarray(muap_classes)[np.asarray(muap_classes) == j])))
                    plt.xlabel("Samples[n]")
                    plt.ylabel("Amplitude(uV)")
                    plt.grid()
                    plt.subplots_adjust(hspace=0.9, wspace=0.3)
                    for k in range(len(muaps)):
                        if muap_classes[k] == j:
                            plt.plot(muaps[k])
                    plt.figure(plot_fig_num+1)
                    plt.subplot(rows, cols, j + 1)
                    plt.title("MU" + str(j + 1) + " : " + str(len(expanded)))
                    plt.xlabel("Samples[n]")
                    plt.ylabel("Amplitude(uV)")
                    plt.grid()
                    plt.subplots_adjust(hspace=0.9, wspace=0.3)
                    for k in range(len(expanded)):
                        plt.plot(expanded[k])

                    plt.figure(plot_fig_num+2)
                    plt.subplot(rows, cols, j+1)
                    plt.title("MU" + str(j + 1))
                    plt.xlabel("Samples[n]")
                    plt.ylabel("Amplitude(uV)")
                    plt.grid()
                    plt.subplots_adjust(hspace=0.9, wspace=0.3)
                    plt.plot(avg_class)

                    plt.figure(plot_fig_num + 3)
                    plt.subplot(rows, cols, j + 1)
                    if amplitude > -sys.maxsize:
                        plt.title("MU" + str(j + 1) + "- A: " + str(int(amplitude)) + ", D: " + "{0:.2f}".format(duration)
                                  + ", Ar: " + str(int(area)) + ", T: " + str(turns) + ", P: " + str(phases)
                                  + ', R:' + "{0:.2f}".format(rise_time))
                    else:
                        plt.title("MU" + str(j + 1) + "- Amplitude: N/A")
                    plt.xlabel("Samples[n]")
                    plt.ylabel("Amplitude(uV)")
                    plt.grid()
                    plt.subplots_adjust(hspace=0.9, wspace=0.3)
                    plt.plot(baseline_corrected, label='Average MUAP')
                    if amplitude > -sys.maxsize:
                        plt.plot(amplitude_peaks, np.asarray(baseline_corrected)[amplitude_peaks], 'bx', label='Peaks of Amplitude')
                    if duration > -sys.maxsize:
                        plt.plot(points[0], np.asarray(baseline_corrected)[points[0]], 'rx', label='BEP')
                        plt.plot(points[1], np.asarray(baseline_corrected)[points[1]], 'gx', label='EEP')

            if plot_fig_num > 0:
                if plot_mode == 'debug':
                    plt.show()
                elif plot_mode == 'interactive':
                    plt.pause(2)
        mu_expanded_signals.append(muap_expanded_signal)
        mu_average_signals.append(muap_average_signal)
        mu_baseline_corrected_signals.append(muap_baseline_corrected_signal)
        if len(muap_extracted_feature) == 8:
            validated = True
            for s in range(8):
                if len(muap_extracted_feature[s]) != 5:
                    print('Unequal number of features for MUAP No. ' + str(s) + '. Expected 5, Found ' + str(len(muap_extracted_feature[s])))
                    validated = False
                    break
            if validated:
                extracted_features.append(muap_extracted_feature)
                feature_labels.append(labels[i])
            else:
                print('Feature Incomplete')
        else:
            print('MUAP Classes incomplete for dataset. Expected 8, Found ' + str(len(muap_extracted_feature)))


    return extracted_features, feature_labels
