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
def calculate_avg_spectral_amplitude(data, thresh=None):
    fourier, peaks = calculate_spectral_peak(data, thresh)
    return np.average(fourier[peaks])



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
    return autocorr_xdm[len(data) - 1]
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
    data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\'
    # 1. Data Acquisition
    print("LOADING DATA SET")
    urls, labels, label_map = get_dataset(data_base_dir, shuffle=True)

    if "als" in label_map:
        als_patient_label = label_map.index("als")
    elif "neuropathy" in label_map:
        als_patient_label = label_map.index("neuropathy")
    else:
        als_patient_label = -1

    data_filename = 'data.npy'
    header_filename = 'data.hea'
    print('Dataset Loaded - Total: ' + str(len(urls)) + ', Output Classes: ' + str(len(label_map)))

    output_dir = "time_freq_classification_output"

    