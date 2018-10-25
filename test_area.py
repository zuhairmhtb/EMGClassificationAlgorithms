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
    return result
# Calculate zero lag value of autocorrelation of a data
def calculate_zero_lag_autocorrelation(data):
    # Remove sample mean.
    xdm = np.asarray(data) - np.mean(data)
    autocorr_xdm = np.correlate(xdm, xdm, mode='full')
    return autocorr_xdm[int(len(data)/2) - 1]
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

# Test Section Specific modules

if __name__ == "__main__":

    # ------------------------------1. DATA ACQUISITION-------------------------------------------------
    data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\simulated_dataset\\'
    result_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\emg_classification_library\\time_freq_classification_output\\'
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

    data_np = []
    total_frames = 64
    total_samples_per_frame = 4096


    segmented_data = []
    sampling_rates = []
    plot_als = []
    plot_normal = []
    for i in range(len(urls)):
        d = np.load(os.path.join(urls[i], data_filename))
        fs = read_sampling_rate(os.path.join(urls[i], header_filename))
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

    cropped_data = []
    crop_start = 30
    crop_length = 25
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
            plot_normal = [cd, sampling_rates[i], "Healthy Subject", d]

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
    filter_band = 'lowpass'
    filter_range = [1500]
    plot_als = []
    plot_als_b = []
    plot_normal = []
    plot_normal_b = []
    for i in range(len(cropped_data)):
        signal = []
        segmented_signal = []
        for j in range(len(cropped_data[i])):
            signal = signal + list(cropped_data[i][j])
            segmented_signal.append(butter_bandpass_filter(cropped_data[i][j],
                                    filter_range, filter_band, sampling_rates[i], order=2))
        fd = butter_bandpass_filter(np.asarray(signal), filter_range, filter_band, sampling_rates[i], order=2)
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
                    autocorrelation = calculate_autocorrelation(collected_data[i][j][segments[k]])

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


    # 1. Average Spectral Amplitude
    avg_amplitude_table = PrettyTable()
    avg_amplitude_table.field_names = ["SL No.", "Subject Type", "Maximum Amplitude",
                                       "Minimum Amplitude", "Average Amplitude",
                                       "Maximum Frequency", "Minimum Frequency",
                                       "Average Frequency"]
    total_als = 3
    total_normal = 3

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
            fourier, peaks = calculate_spectral_peak(segmented_filtered_data[i][j])
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
    spectral_peak_table_path = os.path.join(result_base_dir, "simulated_signal_pso_knn_spectral_peaks_table.html")
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
            mf.append(calculate_mean_frequency(d[j], sampling_rates[i]))
            zl.append(calculate_zero_lag_autocorrelation(d[j]))
            zc.append(calculate_zero_crossing_rate(d[j]))
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




    # ------------------------------3. CLASSIFICATION-------------------------------------------------


    n_neighbors = 1

    classification_feature_labels = ["Average Spectral Amplitude", "Mean Frequency",
                                     "Zero Lag", "Zero Crossing rate"]
    classification_features = [segmented_avg_spectral_amplitudes, mean_frequencies,
                               zero_lag, zero_crossing]

    classification_inputs = []
    training_accuracies = []

    test_predictions = []

    total_iterations = 20

    data_size = [(i+1)/total_iterations for i in range(total_iterations)]

    total_train_accuracy = []
    total_test_accuracy = []
    total_specifity = []
    total_sensitivity = []
    total_input_data = []
    total_neighbors = []
    for iter in range(total_iterations):
        print("=================Classification Iteration No. " + str(iter+1)  + "======================\n")
        feature_train_accuracy = []
        feature_test_accuracy = []
        feature_specifity = []
        feature_sensitivity = []
        feature_input = []
        feature_neighbor = []
        for i in range(len(classification_features)):
            print("------Feature: " + str(classification_feature_labels[i]) + "--------------")
            feature_input.append(int(len(classification_features[i])*data_size[iter]))
            print("Input size: " + str(feature_input[i]))
            features = np.asarray(classification_features[i])[0:feature_input[i], :]
            lab = labels[0:feature_input[i]]
            X_train, X_test, y_train, y_test = train_test_split(features, lab, test_size=0.2,
                                                                shuffle=True)
            for t in range(len(label_map)):
                if not (t in y_test):
                    for u in range(len(y_train)):
                        if y_train[u] == t:
                            tmp_lab = y_train[u]
                            tmp_dat = X_train[u]
                            y_train[u] = y_test[0]
                            X_train[u] = X_test[0]
                            y_test[0] = tmp_lab
                            X_test[0] = tmp_dat
                            break

            print("Train data shape: " + str(X_train.shape))
            print("Test data shape: " + str(X_test.shape))
            classifier_neighbor_range = [1]

            if len(classifier_neighbor_range) > 1:
                pso = PSO(knn_optimize, [classifier_neighbor_range[1]], [classifier_neighbor_range[0]], fitness_minimize=False, cost_function_args=(X_train, y_train),
                      verbose=False, ndview=False, max_iteration=50)
                knn_particles, knn_global_best, knn_best_costs = pso.run()
                classification_inputs.append([X_train, X_test, y_train, y_test])
                print("Best Neighbors for classification feature: " + classification_feature_labels[i] + ": " + str(
                    knn_global_best["position"][0]))

                n_neighbors = int(knn_global_best["position"][0])
                classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
                feature_neighbor.append(n_neighbors)
                classifier.fit(X_train, y_train)
                training_accuracies.append(float("{0:.2f}".format(knn_global_best["cost"]*100)))
                ta = float("{0:.2f}".format(knn_global_best["cost"]*100))

            else:
                n_neighbors = classifier_neighbor_range[-1]
                folds = 0.1
                feature_neighbor.append(n_neighbors)
                t_size = int(folds*len(X_train))
                y_train = np.asarray(y_train)
                y_test = np.asarray(y_test)
                avg_acc = 0
                avg_total = 0
                for n in range(0, int(len(X_train)/t_size), t_size):
                    x1 = X_train[n:n+t_size, :]
                    y1 = y_train[n:n+t_size]
                    x2 = np.concatenate((X_train[0:n, :], X_train[n+t_size:, :]))
                    y2 = np.concatenate((y_train[0:n], y_train[n+t_size:]))
                    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
                    classifier.fit(x2, y2)
                    acc = classifier.score(x1, y1)
                    avg_acc += acc
                    avg_total += 1
                training_accuracies.append(float("{0:.2f}".format((avg_acc/avg_total)*100)))
                ta = float("{0:.2f}".format((avg_acc/avg_total)*100))
                classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
                classifier.fit(X_train, y_train)
            print("Classifying with " + str(n_neighbors) + " Neighbors")
            predictions = classifier.predict(X_test)
            test_predictions.append(predictions)
            acc = 0
            specifity = 0
            total_als = 0
            sensitivity = 0
            total_other = 0
            for a in range(len(predictions)):
                if predictions[a] == lab[a]:
                    acc += 1
                    if predictions[a] == als_patient_label:
                        specifity += 1
                    else:
                        sensitivity += 1
                if lab[a] == als_patient_label:
                    total_als += 1
                else:
                    total_other += 1
            specifity = (specifity/total_als) * 100
            sensitivity = (sensitivity/total_other) * 100
            print('Predictions: ' + str(predictions))
            print('Target:' + str(np.asarray(list(lab))))
            print('Train Accuracy: ' + str(ta))
            print("Test Accuracy: " + "{0:.2f}".format((acc/len(predictions))*100))
            print("Specifity: " + "{0:.2f}".format(specifity))
            print("Sensitivity: " + "{0:.2f}".format(sensitivity))
            feature_train_accuracy.append(ta)
            feature_test_accuracy.append((acc/len(predictions))*100)
            feature_specifity.append(specifity)
            feature_sensitivity.append(sensitivity)

        total_train_accuracy.append(feature_train_accuracy)
        total_test_accuracy.append(feature_test_accuracy)
        total_specifity.append(feature_specifity)
        total_sensitivity.append(feature_sensitivity)
        total_input_data.append(feature_input)
        total_neighbors.append(feature_neighbor)

    fig_num = 8
    performance_table_path = result_base_dir + "simulated_signal_pso_knn_average_performance_graph.html"
    if os.path.exists(performance_table_path):
        with open(performance_table_path, 'r') as fp:
            performance_table = from_html_one(fp.read())
    else:
        performance_table = PrettyTable()
        performance_table.field_names = ["SL No.", "Classification Feature", "Average Train Accuracy(%)",
                                         "Average Test Accuracy(%)", "Average Specifity(%)", "Average Sensitivity(%)"]
    for i in range(len(classification_feature_labels)):
        plt.figure(fig_num+i)
        plt.suptitle("Performance Graph for Classification with feature: " + str(classification_feature_labels[i].upper()))

        plt.xlabel("Input Data size[n]")
        plt.ylabel("Performance(%)")
        plt.grid()
        plt.plot(np.asarray(total_input_data)[:, i], np.asarray(total_train_accuracy)[:, i], label="Train Accuracy")
        plt.plot(np.asarray(total_input_data)[:, i], np.asarray(total_test_accuracy)[:, i], label="Test Accuracy")
        plt.plot(np.asarray(total_input_data)[:, i], np.asarray(total_specifity)[:, i], label="Specifity")
        plt.plot(np.asarray(total_input_data)[:, i], np.asarray(total_sensitivity)[:, i], label="Sensitivity")
        plt.plot(np.asarray(total_input_data)[:, i], np.asarray(total_neighbors)[:, i], label="Nearest Neighbors")
        plt.legend()

        performance_table.add_row([i+1, classification_feature_labels[i].upper(),
                                   "{0:.2f}".format(np.average( np.asarray(total_train_accuracy)[:, i])),
                                   "{0:.2f}".format(np.average( np.asarray(total_test_accuracy)[:, i])),
                                   "{0:.2f}".format(np.average(np.asarray(total_specifity)[:, i])),
                                   "{0:.2f}".format(np.average(np.asarray(total_sensitivity)[:, i]))
                                   ])
    plt.show()
    #with open(performance_table_path, 'w') as fp:
        #fp.write(performance_table.get_html_string())

    # ------------------------------4. PERFORMANCE-------------------------------------------------