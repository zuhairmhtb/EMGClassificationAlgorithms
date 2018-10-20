from scipy.signal import find_peaks
import numpy as np
import peakutils, pdb
import matplotlib.pyplot as plt
"""
Available functions:
"Amplitude Difference(Min -ve and Max+ve Peaks)",
"Duration(BEP, EEP)",
"Rectified Area(Integrated over calculated duration)",
"Rise Time(Time difference between Max -ve and preceeding Min +ve Peak)",
"Phases",
"Turns(No. of +ve and -ve Peaks)"]
"""

debug_mode = False
def debug_output(text):
    if debug_mode:
        print(text)
        
def calculate_amplitude_difference(muaps, min_peak_thresh=-1):
    """

    :param muaps: The MUAP Waveforms in a list whose amplitude difference needs to be calculated - list[]
    :return: A list containing the amplitude difference for each MUAP waveform

    Amplitude Difference: Amplitude difference between maximum negative and minimum positive peaks- list[]
    """
    amp_difference = []
    for i in range(len(muaps)):
        muap = np.asarray(muaps[i])
        peak_thresh = min_peak_thresh
        if peak_thresh < 0:
            peak_thresh = np.mean(np.abs(muap))
        peaks, properties = find_peaks(muap, peak_thresh)
        inverse_peaks, inverse_properties = find_peaks(muap*-1, peak_thresh)
        if len(peaks) > 0 and len(inverse_peaks) > 0:
            amp_diff = np.amax(muap[inverse_peaks]) - np.amin(muap[peaks])
        elif len(peaks) == 0 and len(inverse_peaks) > 0:
            amp_diff = np.amax(muap[inverse_peaks]) - 0
        elif len(peaks) > 0 and len(inverse_peaks) == 0:
            amp_diff = 0 - np.amin(muap[peaks])
        else:
            amp_diff = muap[int(len(muap)/2)]

        amp_difference.append(amp_diff)
    return amp_difference
def calculate_waveform_duration(muaps, window_duration):
    """

    :param muaps: The MUAP Waveforms in a list whose duration needs to be calculated - List[],
           window_duration: Total duration(ms) of each MUAP waveform - float
    :return: Duration: A list of containing duration (ms) for each MUAP - List[]
             [BEP, EEP]: A list containing [Begin Extraction, End Extraction] points for each MUAP - List[[BEP,EEP]]

    Duration: Starting from the beginning of the MUAP waveform find the first point where the signal is
    greater than a threshold equal to 1/15 of the amplitude difference. The threshold is allowed to take
    values between 10 and 20 uV. This allows the algorithm to identify the waveform areas
    close to which the MUAP beginning and ending points are expected to be found.
    The duration(ms) of the MUAP is the time interval between MUAP beginning and ending points.
    """
    min_amp_amount = 1/15
    amp_difference = calculate_amplitude_difference(muaps)
    thresh_min_allowed_voltage = 10
    thresh_max_allowed_voltage = 20
    waveform_duration = []
    waveform_endpoints = []
    for i in range(len(muaps)):
        muap = np.asarray(muaps[i])
        amplitude = amp_difference[i]
        bep_index = -1
        eep_index = -1
        for j in range(len(muap)):
            if muap[j] > min_amp_amount * amplitude and thresh_min_allowed_voltage < muap[j] < thresh_max_allowed_voltage:
                bep_index = j
                break
        for j in range(len(muap)-1, -1, -1):
            if muap[j] > min_amp_amount * amplitude and thresh_min_allowed_voltage < muap[j] < thresh_max_allowed_voltage:
                eep_index = j
                break
        if bep_index >= 0 and eep_index < len(muap) and bep_index != eep_index:
            duration = window_duration*(eep_index - bep_index)/len(muap)
        else:
            duration = window_duration
            bep_index = 0
            eep_index = len(muap)-1
        waveform_endpoints.append([bep_index, eep_index])
        waveform_duration.append(duration)

    return waveform_duration, waveform_endpoints
def calculate_rectified_waveform_area(muaps, window_duration):
    """
    :param muaps: List of waveforms of Muscle Unit Action Potential - List[]
           window_duration: Total duration(ms) of each MUAP waveform - float
    :return: rectified_area: Rectified MUAP Area over the calculated waveform duration - List[]

    Rectified Area: Rectified MUAP integrated over the calculated duration.
    """
    _, bep_eep = calculate_waveform_duration(muaps, window_duration)
    rectified_area = []
    for i in range(len(muaps)):
        muap = np.asarray(muaps[i])
        rectified_muap = np.abs(muap)
        area = np.sum(rectified_muap[bep_eep[i][0] : bep_eep[i][1]])
        rectified_area.append(area)
    return rectified_area
def calculate_rise_time(muaps, window_duration, min_peak_thresh=-1):
    """
    :param muaps: List of waveforms of Muscle Unit Action Potential - List[]
           window_duration: Total duration(ms) of each MUAP waveform - float
    :return: Rise Time: Rise Time(ms) for each MUAP waveform - List[]

    Rise Time: Time between maximum negative peak and the preceding minimum positive peak within the duration.
    """
    _, bep_eep = calculate_waveform_duration(muaps, window_duration)
    rise_time = []
    for i in range(len(muaps)):
        muap = np.asarray(muaps[i])[bep_eep[i][0]:bep_eep[i][1]]
        peak_thresh = min_peak_thresh
        if peak_thresh < 0:
            neg_peak_thresh = np.mean(muap*-1)
            peak_thresh = np.mean(muap)

        neg_peaks, _ = find_peaks(muap*-1, neg_peak_thresh)
        min_pos_peak = len(muap)-1
        max_neg_peak = 0
        if len(neg_peaks) > 0:
            max_neg_peak = np.argmax(muap[neg_peaks])

        pos_peaks, _ = find_peaks(muap[:max_neg_peak], peak_thresh)

        if len(pos_peaks) > 0 and np.argmin(muap[pos_peaks]) != max_neg_peak:
            min_pos_peak = np.argmin(muap[pos_peaks])
        elif len(pos_peaks) > 0 and np.argmin(muap[pos_peaks]) == max_neg_peak:
            pos_peaks, _ = find_peaks(muap[max_neg_peak:], peak_thresh)
            if len(pos_peaks) > 0:
                min_pos_peak = np.argmin(muap[pos_peaks])
        rise = window_duration*abs(max_neg_peak - min_pos_peak) / len(muaps[i])
        rise_time.append(rise)
    return rise_time

def calculate_phase(muaps, window_duration):
    """
    :param muaps: List of waveforms of Muscle Unit Action Potential - List[]
           window_duration: Total duration(ms) of each MUAP waveform - float
    :return: Phase: Number of phases for each MUAP waveform - List[]

    Phase: Number of baseline crossings within the duration where amplitude exceeds 25 V, plus one.
    """
    amp_thresh = 25
    phases = []
    _, bep_eep = calculate_waveform_duration(muaps, window_duration)
    for i in range(len(muaps)):
        muap = np.asarray(muaps[i])[bep_eep[i][0]:bep_eep[i][1]]
        baseline = [0] * len(muap)
        last_crossed_above = False
        total_crossings = 0
        for j in range(len(muap)):
            if last_crossed_above:
                if muap[j] < baseline[j] and muap[j] < - amp_thresh:
                    total_crossings += 1
                    last_crossed_above = False
            else:
                if muap[j] > baseline[j] and muap[j] >  amp_thresh:
                    total_crossings += 1
                    last_crossed_above = True
        phases.append(total_crossings)

    return phases
def calculate_turns(muaps, window_duration, min_peak_thresh=-1, trim_muap=False):
    """
    :param muaps: List of waveforms of Muscle Unit Action Potential - List[]
           window_duration: Total duration(ms) of each MUAP waveform - float
    :return: Turns: Number of turns for each MUAP waveform - List[]

    Turns: Number of positive and negative peaks where the differences from the preceding and following
    turn exceed 25 uV
    """
    diff_thresh = 25
    turns = []
    _, bep_eep = calculate_waveform_duration(muaps, window_duration)

    for i in range(len(muaps)):
        if len(np.asarray(muaps[i])[bep_eep[i][0]:bep_eep[i][1]]) > 10 and trim_muap:
            muap = np.asarray(muaps[i])[bep_eep[i][0]:bep_eep[i][1]]
        else:
            muap = np.asarray(muaps[i])
        peak_thresh = min_peak_thresh
        if peak_thresh < 0:
            peak_thresh = np.mean(muap)
        pos_peak, _ = find_peaks(muap, peak_thresh)

        if peak_thresh < 0:
            peak_thresh = np.mean(muap*-1)
        neg_peaks, _ = find_peaks(muap*-1, peak_thresh)

        total_peaks = list(pos_peak) + list(neg_peaks)
        debug_output(total_peaks)
        debug_output(pos_peak)
        debug_output(neg_peaks)
        peak_signs = [1]*len(pos_peak) + [-1]* len(neg_peaks)
        sorted_index = np.argsort(total_peaks)
        total_peaks = np.asarray(total_peaks)[sorted_index]
        peak_signs = np.asarray(peak_signs)[sorted_index]
        total_turns = 0
        for j in range(len(total_peaks)-1):
            if peak_signs[j] != peak_signs[j+1] and abs(muap[total_peaks[j]] - muap[total_peaks[j+1]]) >= diff_thresh:
                total_turns += 1
        turns.append(total_turns)
    return turns




dir = "D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\train\\als\\a01_patient\\N2001A01BB02\\"
a = np.load(dir+"muap_waveforms.npy")
b = np.load(dir+"muap_output_classes.npy")
c = np.load(dir+"muap_superimposition_classes.npy")
d = np.load(dir+"muap_firing_time.npy")
a = np.asarray(a)
c = np.asarray(c)
muap = a[c==0]
#calculate_turns(muap, 6)
