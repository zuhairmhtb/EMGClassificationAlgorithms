"""
This program is an implementation of EMG Signal Classification technique using Time-Frequency Domain
Analysis. It reconstructs the following research methodologies in order to compare the impact of the
classification technique on varying data set:

1. IDENTIFYING THE MOTOR NEURON DISEASE IN EMG SIGNAL USING TIME AND FREQUENCY DOMAIN FEATURES WITH COMPARISON
   (DOI: 10.5121/sipij.2012.3207)

"""
import os, random, pywt, sys, pdb, datetime, collections
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

# Load data from dataset
def get_dataset(data_base_dir, shuffle=True):
    data_type = {}
    total = 0
    urls = []
    labels = []
    label_map = []
    for dt in os.listdir(data_base_dir):
        dt_p = os.path.join(data_base_dir, dt)
        if os.path.isdir(dt_p):
            data_type[dt] = {}

            for disease in os.listdir(dt_p):
                if disease in label_map:
                    label = label_map.index(disease)
                else:
                    label_map.append(disease)
                    label = label_map.index(disease)

                disease_p = os.path.join(dt_p, disease)
                if os.path.isdir(disease_p):
                    data_type[dt][disease] = {}
                    for pat in os.listdir(disease_p):
                        pat_p = os.path.join(disease_p, pat)
                        if os.path.isdir(pat_p):
                            data_type[dt][disease][pat] = {}
                            for rec in os.listdir(pat_p):
                                rec_p = os.path.join(pat_p, rec)
                                if os.path.isdir(rec_p):
                                    data_type[dt][disease][pat][rec] = rec_p
                                    urls.append(rec_p)
                                    labels.append(label)
                                    total += 1

    print(type(labels))
    if shuffle and len(urls) > 0:
        c = list(zip(urls, labels))
        random.shuffle(c)
        urls, labels = zip(*c)

    return urls, labels, label_map

# Extract sampling rate from header file
def read_sampling_rate(path):
    file = open(path, 'r')
    content = file.read().split("\n")
    sampling_rate = float(content[0].split(" ")[2])
    return sampling_rate

# Calculate Discrete Wavelet Transform
def calculate_dwt(data, method='haar', thresholding='soft', level=1, threshold=True):

    if  level<=1:
        (ca, cd) = pywt.dwt(data, method)
        if threshold:
            cat = pywt.threshold(ca, np.std(ca) / 2, thresholding)
            cdt = pywt.threshold(cd, np.std(cd) / 2, thresholding)
            return cat, cdt
        else:
            return ca, cd
    else:
        decs = pywt.wavedec(data, method, level=level)
        if threshold:
            result=[]
            for d in decs:
                result.append(pywt.threshold(d, np.std(d) / 2, thresholding))
            return result
        else:
            return decs

def butter_bandpass(cutoff_freqs, fs, btype, order=5):
    nyq = 0.5 * fs
    for i in range(len(cutoff_freqs)):
        cutoff_freqs[i] = cutoff_freqs[i] / nyq

    b, a = butter(order, cutoff_freqs, btype=btype)
    return b, a

def butter_bandpass_filter(data, cutoff_freqs, btype, fs, order=5):
    b, a = butter_bandpass(cutoff_freqs.copy(), fs, btype, order=order)
    y = lfilter(b, a, np.copy(data))
    return y

# Calculate mean of absolute value of a list of time series
def calculate_mav(data):
    result = [np.mean(np.abs(data[i])) for i in range(len(data))]
    return result

# Calculate average power of a list of time series
def calculate_avp(data):
    result = [(1/(2*len(data[i]) + 1)) * np.sum(np.square(np.abs(data[i]))) for i in range(len(data))]
    return result

# Calculate Standard deviation of a list of time series
def calculate_std(data):
    result = [np.std(data[i]) for i in range(len(data))]
    return result

# Calculate Ratio of Absolute Mean Values between adjacent data of a list of time series
def calculate_ram(data):
    result = []
    for i in range(len(data)-1):
        result.append( np.abs(np.mean(data[i]))/np.abs(np.mean(data[i+1]))  )
    return result

# Crop a signal by duration(ms)
def crop_data(data, fs, crop_duration):
    """

    :param data: The signal that needs to be cropped (Array like)
    :param fs: Sampling rate of the signal (float)
    :param crop_duration: The amount of duration that needs to be kept (ms) float
    :return: Cropped Signal (Array like)
    """
    keep_length = int(fs*crop_duration/1000)
    if keep_length < len(data):
        crop_length = len(data) - keep_length
        crop_start = int(crop_length/2)
        crop_end = crop_length - crop_start
        return data[crop_start:len(data)-crop_end]
    return data

# Calculate Peaks from Magnitude Spectrum/FFT Magnitude of a data
def calculate_spectral_peak(data, thresh=None):
    fourier = np.fft.fft(data)
    fourier = abs(fourier[0:len(data) // 2])

    if thresh is None:
        peaks, props = find_peaks(fourier, height=np.mean(fourier))
        return fourier, peaks
    else:
        peaks, props = find_peaks(fourier, height=thresh)
        return fourier, peaks

# Calculate Average Spectral Amplitude from FFT of a data
def calculate_max_spectral_amplitude(data, thresh=None):
    fourier, peaks = calculate_spectral_peak(data, thresh)
    if len(peaks) > 0:
        return np.amax(fourier[peaks])
    else:
        return 0



# Calculate mean frequency from a signal
def calculate_mean_frequency(data, fs):
    fourier = np.fft.fft(data)
    fourier = abs(fourier[0:len(data) // 2])
    freqs = np.fft.fftfreq(len(data), 1/fs)[0:len(data) // 2]
    return np.sum(fourier*freqs)/np.sum(fourier)

# Calculate autocorrelation of a signal
def calculate_autocorrelation(data):
    result = np.correlate(data, data, mode='full')
    return result[int(result.size / 2):]
# Calculate zero lag value of autocorrelation of a data
def calculate_zero_lag_autocorrelation(data):
    # Remove sample mean.
    xdm = np.asarray(data) - np.mean(data)
    autocorr_xdm = np.correlate(xdm, xdm, mode='full')
    return autocorr_xdm[int(len(data)/2)]
# Calculate Zero Crossing Rate
def calculate_zero_crossing_rate(data):
    signs = []
    for i in range(1, len(data)):
        if data[i] >= 0:
            signs.append(1)
        else:
            signs.append(-1)
    vals = [np.abs(signs[i] - signs[i-1]) for i in range(1, len(signs))]
    return (1/(2*len(data))) * np.sum(vals)

def knn_optimize(x, args):
    classifier = KNeighborsClassifier(n_neighbors=int(x[0]))
    if len(args) == 2:
        X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])
if __name__ == "__main__":
    show_plot = False
    verbose = False
    simulated = False
    if not simulated:
        data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\'
        signal_type = 'Real'
    else:
        data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\simulated_dataset\\'
        signal_type = 'Simulated'
    result_base_dir = "D:\\thesis\\ConvNet\\MyNet\\emg_classification_library\\"
    muscle_location = "Biceps Brachii"
    scale = False

    if scale:
        signal_type += "(Scaled)"

    save_result = True
    #number_folds = 10
    result_path = result_base_dir + "fft_performance_table.html"
    if os.path.exists(result_path):
        with open(result_path, 'r') as fp:
            table = from_html_one(fp.read())
    else:
        table = PrettyTable()
        table.field_names = ["Index No.", "Signal Type", "Muscle Location", "Classifier", "Subject type",
                             "Train data size", "Test data size", "Specifity(%)",
                             "Sensitivity(%)", "Performance accuracy(%)", "Test Accuracy(%)","Classification Date"]

    # 1. Data Acquisition
    print("LOADING DATA SET")
    urls, labels, label_map = get_dataset(data_base_dir, shuffle=True)
    #urls = urls[:10]
    #labels = labels[:10]
    if "als" in label_map:
        als_patient_label = label_map.index("als")
    elif "neuropathy" in label_map:
        als_patient_label = label_map.index("neuropathy")
    else:
        als_patient_label = -1
    # urls = urls[:30]
    # labels = labels[:30]
    data_filename = 'data.npy'
    header_filename = 'data.hea'
    print('Dataset Loaded - Total: ' + str(len(urls)) + ', Output Classes: ' + str(len(label_map)))
    print('Label Map: ' + str(label_map))
    print('Labels:  ' + str(type(labels)))
    print('URLS: ' + str(type(urls)))

    # 2. Data Segmentation
    segmented_data = []
    rms_data = []
    cropped_data = []
    cropped_segmented_data = []
    filtered_data = []
    filtered_segmented_data = []

    avg_spectral_amps = []
    segmented_avg_spectral_amps = []
    mean_freqs = []
    segmented_mean_freqs = []
    zero_lags = []
    segmented_zero_lags = []
    zero_crossing_rates = []
    segmented_zero_crossing_rates = []
    spectral_peaks = []
    segmented_spectral_peaks = []
    autocorrelations = []
    segmented_autocorrelations = []

    sampling_rates = []

    total_frames = 64
    total_samples_per_frame = 4096
    original_data = []
    for i in range(len(urls)):
        original_data.append(np.load(os.path.join(urls[i], data_filename)))
    if scale:
        original_data = preprocessing.scale(original_data)
    for i in range(len(urls)):
        if verbose or True:
            print("Left loading data-Total: " + str(len(urls)-i))
        # Load data as numpy array
        d = original_data[i]
        fs = read_sampling_rate(os.path.join(urls[i], header_filename))

        # Adjust From both end: Crop data if the total number of sample is larger than expected sample and add zero otherwise
        expected_samples = total_frames * total_samples_per_frame
        extra = len(d) - expected_samples
        left = int(abs(extra)/2)
        right = abs(extra) - left
        if extra > 0:
            d = d[left:len(d)-right]
        elif extra < 0:
            d = np.asarray([0]*left + list(d.flatten()) + [0]*right)

        # Segment data into specified number of frames and samples per frame
        segmented_data.append([d[i:i+total_samples_per_frame] for i in range(0, d.shape[0], total_samples_per_frame)])

        # Calculate Moving RMS/Smoothing for the segmented data
        rms_data.append([np.sqrt(np.mean(np.square(segmented_data[i][j]))) for j in range(len(segmented_data[i]))])

        # Crop the data into specified length
        total_cropped_frames = 25
        crop_frame_start_index = 30
        cropped_data.append(rms_data[i][crop_frame_start_index: crop_frame_start_index+total_cropped_frames])
        cropped_segmented_data.append(segmented_data[i][crop_frame_start_index: crop_frame_start_index+total_cropped_frames])

        # Apply Low Pass filter to specific frame
        pass_type = 'lowpass'
        pass_band = [1500]  # 1500Hz
        filtered_data.append(butter_bandpass_filter(cropped_data[i], pass_band, pass_type, fs, order=2))
        filtered_segmented_data.append([butter_bandpass_filter(cropped_segmented_data[i][j], pass_band, pass_type, fs, order=2)
                              for j in range(len(cropped_segmented_data[i]))])

        fs = pass_band[0]
        # Extract features from the filtered data frames

        # Spectral Peaks

        spectral_peaks.append(calculate_spectral_peak(filtered_data[i]))
        segmented_spectral_peaks.append([calculate_spectral_peak(filtered_segmented_data[i][j])
                                         for j in range(len(filtered_segmented_data[i]))])
        if len(spectral_peaks[i]) > 0:

            avg_amplitude = calculate_max_spectral_amplitude(filtered_data[i])
        else:
            avg_amplitude = 0
        segmented_avg_amplitude = []
        for j in range(len(segmented_spectral_peaks[i])):
            if len(segmented_spectral_peaks[i][j]) > 0:
                segmented_avg_amplitude.append(calculate_max_spectral_amplitude(filtered_segmented_data[i][j]))
            else:
                segmented_avg_amplitude.append(0)
        avg_spectral_amps.append(avg_amplitude)
        segmented_avg_spectral_amps.append(segmented_avg_amplitude)
        print("Patient type: " + str(label_map[labels[i]]))
        print("Max Spectral Amplitude: " + str(segmented_avg_amplitude))
        if verbose:
            print("PATIENT TYPE: " + str(labels[i]))
            print("Average Spectral Amplitude: " + str(avg_spectral_amps))
            print("Segmented Spectral Amplitude(" + str(len(segmented_avg_spectral_amps[i])) + "): ")

        # Mean Frequency

        mean_freqs.append(calculate_mean_frequency(filtered_data[i], fs))

        smf = [calculate_mean_frequency(filtered_segmented_data[i][j], fs)
                                     for j in range(len(filtered_segmented_data[i]))]
        segmented_mean_freqs.append(smf)
        print("Mean frequency: " + str(smf))
        if verbose:
            print("Mean Frequency: " + str(mean_freqs))
            print(
                "Segmented Mean Frequency(" + str(len(segmented_mean_freqs[i])) + "): " + str(segmented_mean_freqs[i]))

        # Autocorrelation

        autocorrelations.append(calculate_autocorrelation(filtered_data[i]))
        sac = [calculate_autocorrelation(filtered_segmented_data[i][j])
                                     for j in range(len(filtered_segmented_data[i]))]
        segmented_autocorrelations.append(sac)


        if verbose:
            print("Autocorrelation: " + str(autocorrelations))
            print(
                "Segmented Autocorrelation(" + str(len(segmented_autocorrelations[i])) + "): " + str(segmented_autocorrelations[i]))

        # Autocorrelation - Zero lag
        zero_lags.append(calculate_zero_lag_autocorrelation(filtered_data[i]))
        szl = [calculate_zero_lag_autocorrelation(filtered_segmented_data[i][j])
                                     for j in range(len(filtered_segmented_data[i]))]
        segmented_zero_lags.append(szl)
        print('Segmented zero lag: ' + str(szl))
        #print("Segmented zero lag: " + str(szl))
        if verbose:
            print("Zero Lag: " + str(zero_lags))
            print(
                "Segmented Zero Lag(" + str(len(segmented_zero_lags[i])) + "): " + str(segmented_zero_lags[i]))

        # Zero Crossing rate
        zero_crossing_rates.append(calculate_zero_crossing_rate(filtered_data[i]))
        zcr = [calculate_zero_crossing_rate(filtered_segmented_data[i][j])
                                     for j in range(len(filtered_segmented_data[i]))]
        segmented_zero_crossing_rates.append(zcr)
        print("Segmented zero crossing rate: " + str(zcr))
        if verbose:
            print("Zero crossing rate: " + str(zero_crossing_rates))
            print(
                "Segmented Zero Crossing rate(" + str(len(segmented_zero_crossing_rates[i])) + "): " + str(segmented_zero_crossing_rates[i]))

        if show_plot:
            plt.figure(1)
            plt.subplot(4, 2, 1)
            plt.title("Segmented data")
            s = []
            for j in range(len(segmented_data[i])):
                s = s + list(segmented_data[i][j])
            plt.plot(s)
            plt.grid()
            plt.subplot(4, 2, 2)
            plt.title("RMS Data")
            plt.plot(rms_data[i])
            plt.grid()
            plt.subplot(4, 2, 3)
            plt.title("Cropped data")
            plt.plot(cropped_data[i])
            plt.grid()
            plt.subplot(4, 2, 4)
            plt.title("Cropped Segmented data")
            s = []
            for j in range(len(cropped_segmented_data[i])):
                s = s + list(cropped_segmented_data[i][j])
            plt.plot(s)
            plt.grid()
            plt.subplot(4, 2, 5)
            plt.title("Filtered data")
            plt.plot(filtered_data[i])
            plt.grid()
            plt.subplot(4, 2, 6)
            plt.title("Filtered Segmented data")
            s = []
            for j in range(len(filtered_segmented_data[i])):
                s = s + list(filtered_segmented_data[i][j])
            plt.plot(s)
            plt.grid()
            plt.show()


    # Perform Classification
    classification_feature_labels = ["Average Spectral Amplitude", "Mean Frequency",
                                     "Zero Lag", "Zero Crossing rate"]
    classification_features = [segmented_avg_spectral_amps, segmented_mean_freqs,
                               segmented_zero_lags, segmented_zero_crossing_rates]

    classifiers = [KNeighborsClassifier(n_neighbors=1) for _ in range(len(classification_features))]
    classification_inputs = []
    training_accuracies = []
    for j in range(len(classification_features)):
       # if scale:
        #    id = preprocessing.scale(np.asarray(classification_features[j]))
       # else:
        #    id = np.asarray(classification_features[j])

        X_train, X_test, y_train, y_test = train_test_split(classification_features[j], labels, test_size=0.2, shuffle=True)
        pso = PSO(knn_optimize, [20], [2], fitness_minimize=False, cost_function_args=(X_train, y_train),
                  verbose=False, ndview=False)
        knn_particles, knn_global_best, knn_best_costs = pso.run()
        classification_inputs.append([X_train, X_test, y_train, y_test])
        print("Best Neighbors for classification feature: " + classification_feature_labels[j] + ": " + str(knn_global_best["position"][0]))
        classifiers[j] = KNeighborsClassifier(n_neighbors=int(knn_global_best["position"][0]))
        training_accuracies.append("{0:.2f}".format(knn_global_best["cost"] * 100))

    classification_accuracies = []
    classification_predictions = []
    for i in range(len(classification_features)):
        classification_inputs[i][3] = np.asarray(classification_inputs[i][3])
        if verbose:
            print("------------------------------------------------------------")
            print("Classifying For " + str(classification_feature_labels[i]))
        if verbose:
            print("Classification Input Size: " + str(np.asarray(classification_features[i]).shape))
            print("Classification Label Size: " + str(len(labels)))
            print("Train data size: " + str(classification_inputs[i][0].shape))
            print("Train label size: " + str(classification_inputs[i][2]))
            print("Test data size: " + str(classification_inputs[i][1].shape))
            print("Test label size: " + str(len(classification_inputs[i][3])))

        classifiers[i].fit(classification_inputs[i][0], classification_inputs[i][2])
        acc = classifiers[i].score(classification_inputs[i][1], classification_inputs[i][3])
        classification_accuracies.append(float("{0:.2f}".format(acc*100)))
        predictions = classifiers[i].predict(classification_inputs[i][1])
        classification_predictions.append(predictions)
        print("Feature: " + str(classification_feature_labels[i]))
        print("Prediction: " + str(predictions))
        print("Target: " + str(classification_inputs[i][3]))
        if verbose:
            print("Accuracy: " + "{0:.2f}".format(acc*100))
            print("------------------------------------------------------------")

    # Calculate performance of classification
    specifity = []
    sensitivity = []

    for i in range(len(label_map)):
        sp = []
        se = []
        for data in range(len(classification_predictions)):
            total_positive = 0
            total_negative = 0
            correct_positive = 0
            correct_negative = 0
            for predicted in range(len(classification_predictions[data])):
                if labels[predicted] == i:
                    total_positive += 1
                    if labels[predicted] == classification_predictions[data][predicted]:
                        correct_positive += 1
                else:
                    total_negative += 1
                    if labels[predicted] == classification_predictions[data][predicted]:
                        correct_negative += 1
            if total_positive > 0:
                sp.append(float("{0:.2f}".format(correct_positive*100/total_positive)))
            else:
                sp.append(0)
            if total_negative > 0:
                se.append(float("{0:.2f}".format(correct_negative * 100 / total_negative)))
            else:
                se.append(0)
        specifity.append(sp)
        sensitivity.append(se)
    for i in range(len(label_map)):
        print("-------------------------------------------------------------")
        print("Calculating Performance for " + label_map[i])
        for j in range(len(classification_predictions)):
            print("Feature(%): " + str(classification_feature_labels[j]))
            print("Sensitivity(%): " + str(sensitivity[i][j]))
            print("Specifity(%): " + str(specifity[i][j]))
            print("Accuracy(%): " + str(classification_accuracies[j]))
            print("Training Accuracy(%): " + str(training_accuracies[j]))

            # Save the data
            if label_map[i].lower() == "als" or label_map[i].lower() == "neuropathy":
                l = "Neuropathy"
            elif label_map[i].lower() == "other" or label_map[i].lower() == "normal":
                l = "Normal/Healthy"
            elif label_map[i].lower() == "myopathy":
                l = "Myopathy"
            if save_result:
                # table.field_names = ["Index No.", "Signal Type", "Muscle Location", "Classifier", "Subject type",
                #                    "Train data size", "Test data size", "Subject's test data size", "Accuracy(%)", "Classification Date"]
                current_index = len(table._rows) + 1
                now = datetime.datetime.now()
                print([current_index, signal_type, muscle_location, "KNN(neighbors=" + str(classifiers[j].n_neighbors) + ")", l,
                       len(classification_inputs[j][2]), len(classification_inputs[j][3]),
                       specifity[i][j], sensitivity[i][j], classification_accuracies[j], training_accuracies[j], now.strftime("%Y-%m-%d %H:%M")])
                table.add_row([current_index, signal_type, muscle_location, "KNN(neighbors=" + str(classifiers[j].n_neighbors) + ")", l,
                               len(classification_inputs[j][2]), len(classification_inputs[j][3]), specifity[i][j], sensitivity[i][j], classification_accuracies[j],
                               training_accuracies[j], now.strftime("%Y-%m-%d %H:%M")])

        print("-------------------------------------------------------------")
    if save_result:
        with open(result_path, 'w') as fp:
            fp.write(table.get_html_string())

