import numpy as np
from python_speech_features import mfcc
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
from scipy.signal import spectrogram, find_peaks, butter, lfilter
import pywt
import matplotlib.pyplot as plt
"""
Available Time Domain functions:
1. Mean Absolute Deviation (MAD/MAV)
2. Root Mean Square(RMS)
3. Absolute Value(AV)
4. Standard Deviation(STD)
5. Variance(VR)
6. Approximate Entropy(AE)
7. Average Amplitude Change(AAC)
8. Auto-regressive Coefficients(AC)
9. Box Counting Dimension(BCD)
10. Capstral Coefficients(CC)
11. Difference of Absolute Standard Deviation(DASD)
12. Detrended Fluctuation Analysis(DFA)
13. Higuchi's Fractal Dimension(HFD)
14. Histogram(H)
15. Integral Absolute Value(IAV)
16. Kurtosis(K)
17. Log Detector(LD)
18. Slope of Mean Absolute Value(SMAV)
19. Entropy(Ent)
20. Zero Crossing Rate(ZCR)
21. Discrete Wavelet Transform(DWT)
22. Butterworth Filter(Butterpass)
23. Average Power(AVP)
24. Ratio of Absolute Mean Value(RAM)
25. Crop data
26. Autocorrelation Coefficients
27. Zero Lag of Autocorrelation
"""
"""
Available Frequency Domain Features:
1. Mean Frequency(MNF)
2. Median Frequency(MDF)
3. Average Maximum Density(AMD)
4. Variance of Maximum Frequency(VMF)
5. Mean Power(MNP)
6. Peak Frequency(PKF)
7. Total Power(TOP)
8. Variance of Central Frequency(VCF)
9. Number of Peaks(NPF)
10. Spectral Peak(SP)
11. Average Spectral Amplitude(ASA)
"""
# Calculate mean of absolute value of a list of time series
def calculate_mav(data):
    result = [np.mean(np.abs(data[i])) for i in range(len(data))]
    return result

def calculate_mav_single(data):
    return np.mean(np.abs(data))

def calculate_rms(signal):
    return np.sqrt(np.mean(np.asarray(signal)**2))
def calculate_av(signal):
    return np.abs(signal)
# Calculate Standard deviation of a list of time series
def calculate_std(data):
    result = [np.std(data[i]) for i in range(len(data))]
    return result
def calculate_std_single(data):
    return np.std(data)
def calculate_variance(signal):
    return np.var(signal)
def calculate_wl(fs):
    return 1/fs

# Source: https://gist.github.com/f00-/a835909ffd15b9927820d175a48dee41
def calculate_ae(signal, data_run_coeff=2, filtering_level=3):
    U = signal
    m = data_run_coeff
    r = filtering_level
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)


    return abs(_phi(m + 1) - _phi(m))
def calculate_aac(signal):
    amp_change = [signal[i+1] - signal[i] for i in range(len(signal)-1)]
    return np.average(amp_change)

def calculate_cc(signal, fs, window_time=6, nceps=13):
    window_length = int(fs*(window_time/1000))
    return mfcc(signal, fs, winlen=window_length, numcep=nceps)
def calculate_smad(signal, fs, window):
    window_length = int((fs*window)/1000)
    mads = []
    for i in range(0, len(signal)-window_length, window_length):
        mads.append(calculate_mav(signal[i:i+window_length]))
    slopes = [mads[i+1]-mads[i] for i in range(len(mads)-1)]
    return np.mean(slopes)
def calculate_ent(x, modified=False):
    ent = 0.0
    if not modified:
        tot = np.sum(x ** 2)
        ent = np.sum((x ** 2 / tot) * np.log10(x ** 2 / tot))
        # for i in range(len(x)):
        #   quo = x[i] ** 2 /tot
        #  ent += (quo * np.log10(quo))
    else:
        m = np.mean(x)
        for i in range(len(x)):
            quo = np.abs(x[i] - m)
            ent += (quo * np.log10(quo))
    return -ent

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

# Calculate mean frequency from a signal
def calculate_mean_frequency(data, fs):
    fourier = np.fft.fft(data)
    fourier = abs(fourier[0:len(data) // 2])
    freqs = np.fft.fftfreq(len(data), 1/fs)[0:len(data) // 2]
    return np.sum(fourier*freqs)/np.sum(fourier)

def calculate_mdf(intensity, frequency):
    sorted = np.argsort(intensity)
    sorted_intensity = np.asarray(intensity)[sorted]
    sorted_frequency = np.asarray(frequency)[sorted]
    return sorted_frequency[int(len(sorted_frequency)/2)]
def calculate_amd(density, time):
    max_densities = []
    for i in range(len(time)):
        max_densities.append(np.amax(density[:, i]))
    return np.average(max_densities)
def calculate_vmf(density, frequency, time):
    max_freqs = []
    for i in range(len(time)):
        max_freqs.append(frequency[np.argmax(density[:, i])])
    return np.var(max_freqs)
def calculate_mnp(power, time):
    means = []
    for i in range(len(time)):
        means.append(np.mean(power[:, i]))
    return np.mean(means)

def calculate_vcf(power, frequency, time):
    cfs = []
    for i in range(len(time)):
        max_pow = np.argmax(power[:, i])
        max_freq = frequency[max_pow]
        min_pow = np.argmin(power[:, i])
        min_freq = frequency[min_pow]
        cfs.append(int((max_freq+min_freq)/2))
    return np.var(cfs)
def calculate_pkf(power, frequency, time):
    frequencies = [0 for _ in range(len(frequency))]
    for i in range(len(time)):
        max_pow = np.argmax(power[:, i])
        frequencies[max_pow] += 1
    return frequency[np.argmax(frequencies)]

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    det_peaks = local_max ^ eroded_background
    dp = det_peaks.flatten()
    p = 0
    for i in range(len(dp)):
        if dp[i] == 1:
            p += 1
    return p, det_peaks
def calculate_npf(power, min_height, min_distance):
    power_float = np.asarray(power)/np.amax(np.abs(power))
    peaks = []
    dp = np.zeros(power.shape, np.int64)
    for i in range(power.shape[0]):
        p, prop = find_peaks(power[i, :], height=min_height, distance=min_distance)
        if len(p) > 0:
            dp[i, p] = 1
        peaks.append(p)


    return peaks, dp

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
# Calculate average power of a list of time series
def calculate_avp(data):
    result = [(1/(2*len(data[i]) + 1)) * np.sum(np.square(np.abs(data[i]))) for i in range(len(data))]
    return result
def calculate_avp_single(data):
    return (1/(2*len(data) + 1)) * np.sum(np.square(np.abs(data)))

# Calculate Ratio of Absolute Mean Values between adjacent data of a list of time series
def calculate_ram(data):
    result = []
    for i in range(len(data)-1):
        result.append( np.abs(np.mean(data[i]))/np.abs(np.mean(data[i+1]))  )
    return result
def calculate_ram_single(current_data, next_data):
    return np.abs(np.mean(current_data))/np.abs(np.mean(next_data))
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

dir = "dataset\\train\\als\\a01_patient\\N2001A01BB02\\"
