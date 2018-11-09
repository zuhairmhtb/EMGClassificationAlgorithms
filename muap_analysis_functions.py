from scipy.signal import find_peaks, correlate
import numpy as np
import math, queue, sys

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

    Amplitude Difference: Amplitude difference between maximum positive and minimum negative peaks- list[]
    """
    amp_difference = []
    amp_peak_indices = []
    for i in range(len(muaps)):
        muap = np.asarray(muaps[i])
        peak_thresh = min_peak_thresh
        if peak_thresh < 0:
            peak_thresh = np.mean(muap[muap > 0])
        peaks, properties = find_peaks(muap, height=peak_thresh)
        peak_thresh = min_peak_thresh

        if peak_thresh < 0:
            peak_thresh = np.mean((muap*-1)[muap*(-1) > 0])
        inverse_peaks, inverse_properties = find_peaks(muap*-1, height=peak_thresh)
        if len(peaks) > 0 and len(inverse_peaks) > 0:
            amp_diff = np.abs(np.amin(muap[inverse_peaks]) -np.amax(muap[peaks]))
            pos_peak_val = -sys.maxsize
            pos_peak = -1
            for j in range(len(peaks)):
                if muap[peaks[j]] > pos_peak_val:
                    pos_peak_val = muap[peaks[j]]
                    pos_peak = peaks[j]

            neg_peak_val = sys.maxsize
            neg_peak = -1
            for j in range(len(inverse_peaks)):
                if muap[inverse_peaks[j]] < neg_peak_val:
                    neg_peak_val = muap[inverse_peaks[j]]
                    neg_peak = inverse_peaks[j]
        elif len(peaks) == 0 and len(inverse_peaks) > 0:
            amp_diff = np.abs(np.amin(muap[inverse_peaks]))
            pos_peak = 0
            neg_peak_val = sys.maxsize
            neg_peak = -1
            for j in range(len(inverse_peaks)):
                if muap[inverse_peaks[j]] < neg_peak_val:
                    neg_peak_val = muap[inverse_peaks[j]]
                    neg_peak = inverse_peaks[j]
        elif len(peaks) > 0 and len(inverse_peaks) == 0:
            amp_diff = np.amax(muap[peaks])
            pos_peak_val = -sys.maxsize
            pos_peak = -1
            for j in range(len(peaks)):
                if muap[peaks[j]] > pos_peak_val:
                    pos_peak_val = muap[peaks[j]]
                    pos_peak = peaks[j]
            neg_peak = 0
        else:
            amp_diff = muap[int(len(muap)/2)]
            pos_peak = int(len(muap)/2)
            neg_peak = 0
        amp_peak_indices.append([pos_peak, neg_peak])
        amp_difference.append(amp_diff)
    return amp_difference, amp_peak_indices
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
    min_amp_amount = 1/5
    amp_difference, _ = calculate_amplitude_difference(muaps)
    thresh_min_allowed_voltage = 10
    thresh_max_allowed_voltage = 20
    waveform_duration = []
    waveform_endpoints = []
    for i in range(len(muaps)):
        muap = np.asarray(muaps[i])
        amplitude = amp_difference[i]
        bep_index = -1
        eep_index = len(muap)+1
        thresh = min_amp_amount * amplitude
        for j in range(int(len(muap)/2)):
            if muap[j] > thresh:
                bep_index = j
                break
        for j in range(len(muap)-1, int(len(muap)/2), -1):
            if muap[j] > thresh:
                eep_index = j
                break
        if bep_index >= 0 and eep_index < len(muap) and bep_index != eep_index:
            # The point closest to the baseline starting from the bep_index for 1ms window is the beginning point
            # and the point closest to the baseline starting from eep_index for 1ms window is the ending point

            window_sample_length = int(len(muap)*1/window_duration)
            begin_point = bep_index
            begin_diff = np.abs(np.abs(muap[begin_point])-thresh_max_allowed_voltage)


            for j in range(bep_index, 0, -window_sample_length):
                if j - window_sample_length >= 0:
                    window_start = j - window_sample_length
                else:
                    break
                current_min_index = np.argmin(np.abs(np.abs(muap[window_start:j+1])-thresh_max_allowed_voltage)) + window_start
                if np.abs(np.abs(muap[current_min_index])-thresh_max_allowed_voltage) < begin_diff:
                    begin_diff = np.abs(np.abs(muap[current_min_index])-thresh_max_allowed_voltage)
                    begin_point = current_min_index

            end_point = eep_index
            end_diff = np.abs(np.abs(muap[end_point])-thresh_max_allowed_voltage)

            for j in range(eep_index, len(muap)-1, window_sample_length):
                if j + window_sample_length <= len(muap):
                    window_end = j + window_sample_length
                    print("Current MUAP Length: " + str(len(muap)) + ", win_start: " + str(j)
                          + ", win_end: " + str(window_end))
                    current_min_index = np.argmin(
                        np.abs(np.abs(muap[j:window_end]) - thresh_max_allowed_voltage)) + j

                    if np.abs(np.abs(muap[current_min_index]) - thresh_max_allowed_voltage) < end_diff:
                        end_diff = np.abs(np.abs(muap[current_min_index]) - thresh_max_allowed_voltage)
                        end_point = current_min_index
                else:
                    break

            duration = window_duration*(end_point - begin_point)/len(muap)
        else:
            duration = window_duration
            begin_point = 0
            end_point = len(muap)-1
        waveform_endpoints.append([begin_point, end_point])
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

        neg_peaks, _ = find_peaks(muap*-1, height=neg_peak_thresh)
        min_pos_peak = len(muap)-1
        max_neg_peak = 0
        if len(neg_peaks) > 0:
            max_neg_peak = np.argmax(muap[neg_peaks])

        pos_peaks, _ = find_peaks(muap[:max_neg_peak], height=peak_thresh)

        if len(pos_peaks) > 0 and np.argmin(muap[pos_peaks]) != max_neg_peak:
            min_pos_peak = np.argmin(muap[pos_peaks])
        elif len(pos_peaks) > 0 and np.argmin(muap[pos_peaks]) == max_neg_peak:
            pos_peaks, _ = find_peaks(muap[max_neg_peak:], height=peak_thresh)
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
        total_crossings = 0
        for j in range(1, len(muap)):
            if muap[j-1] > amp_thresh:
                region_prev = 1
            elif muap[j-1] < -amp_thresh:
                region_prev = -1
            else:
                region_prev = 0
            if muap[j] > amp_thresh:
                region_cur = 1
            elif muap[j] < -amp_thresh:
                region_cur = -1
            else:
                region_cur = 0
            if (region_cur == 1 and (region_prev == 0 or region_prev == -1)) or (region_cur == -1 and (region_prev == 0 or region_prev == 1)):
                total_crossings += 1
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
            peak_thresh = np.mean(np.abs(muap))
        pos_peak, _ = find_peaks(muap, height=peak_thresh)

        if peak_thresh < 0:
            peak_thresh = np.mean(np.abs(muap))
        neg_peaks, _ = find_peaks(muap*-1, height=peak_thresh)

        total_peaks = list(pos_peak) + list(neg_peaks)
        t = 0
        total_peaks = np.sort(total_peaks)
        for j in range(1, len(total_peaks)):
            if abs(total_peaks[j] - total_peaks[j-1]) > diff_thresh:
                t += 1

        turns.append(t)
    return turns

# Generate Potential MUAP Waveforms and their firing time from Signal
def muap_an_get_segmentation_const_window(data, fs, window_ms=6, mav_coeff=30, peak_amp_increase_amount=40,
                                          peak_amp_increase_duration=0.1, window_length=-1, verbose=True):
    if window_length <= 0:
        window_samples = int(math.ceil((fs*window_ms)/1000))  # Calculate Segmentation widnow samples
    else:
        window_samples = window_length
    max_val = np.amax(data)  # Calculate Maximum Amplitude of the signal
    mav = (mav_coeff/len(data)) * np.sum(np.abs(data))  # Calculate Mean Absolute Value(MAV) of the signal
    signal_duration = math.ceil(len(data)/fs)  # Calculate duration of the signal in seconds to set threshold coefficient
    if signal_duration <= 0:
        signal_duration = 5 # Default coefficient for 5s EMG signal

    # If the maximum value of the signal is greater than MAV
    if max_val > mav:
        thresh = (signal_duration/len(data)) * np.sum(np.abs(data))  # Set the peak detection min threshold as MAV
    else:
        thresh = max_val/signal_duration  # Set the peak detection min threshold as the maximum amplitude

    # calculate the number of sample points in each segmented peak(Candidate MUAP) for which the amplitude increase
    # of the candidate MUAP waveform must be greater than the specified threshold voltage
    peak_amp_increase_length = math.ceil(fs/(1000*peak_amp_increase_duration))
    if verbose:
        print('Mean Amplitude value: ' + str(mav))
        print('Max val: ' + str(max_val))

    # Find peaks in the signal to obtain candidate MUAP waveform. Minimum distance between the peaks must be atleast
    # one third of the segmentation window length
    peaks = find_peaks(data, thresh, distance=int(window_samples/3))[0]
    potential_muap = []  # Stores the candidate MUAP waveforms

    # For each detected peak in the signal
    for i in range(len(peaks)):
        current_peak_index = peaks[i]  # Get the current Peak index

        # if the peak satisfies the criterion of a valid MUAP
        if current_peak_index - peak_amp_increase_length > 0 and data[current_peak_index]-data[current_peak_index-peak_amp_increase_length] > peak_amp_increase_amount:
            if current_peak_index-int(window_samples/2) > 0 and current_peak_index+int(window_samples/2) < len(data):

                # Extract the candidate MUAP waveform of the specified segmentation length and peak centered.
                current_max_amp = data[current_peak_index]  # Set current maximum amplitude of the window equal to peak

                # For each point in the window(Candidate MUAP waveform)
                for j in range(current_peak_index-int(window_samples/2), current_peak_index+int(window_samples/2)):
                    # If the point represents a peak and is greater than current maximum amplitude
                    if j in peaks and data[j] > current_max_amp:
                        current_max_amp = data[j]  # Set it's amplitude as current max peak amplitude in the window
                        current_peak_index = j  # Set its index as the index of the peak of the window's candidate MUAP
                # If the candidate MUAP waveform is not already in the list
                if not (current_peak_index in potential_muap):
                    # Add the index of peak of the waveform to the list of potential candidate MUAP waveforms
                    potential_muap.append(current_peak_index)
    muap_waveforms = []  # Stores the validated Candidate MUAP waveforms
    muap_firing_times = []  # Stores firing time of the Candidate MUAP waveforms

    # For each MUAP peak index in the list of potential Candidate MUAP waveforms
    for i in range(len(potential_muap)):
        # Add the waveform to the list of candidate MUAP waveform
        muap_waveforms.append(np.asarray(
            data[potential_muap[i]-int(window_samples/2):potential_muap[i]+int(window_samples/2)+1]
        ))
        # Add index/time of the peak as firing time of the MUAP waveform
        muap_firing_times.append(potential_muap[i])
    # Return the identified MUAP waveform peak indices, waveforms and their corresponding firing time
    return potential_muap, muap_waveforms, muap_firing_times

def custom_muap_sofm_classification(muaps, muap_firing_time, muap_firing_table, muap_size=[8, 120], lvq2=False,
                                    init_weight=0.0001, init_mid_weight_const=0.1, g=1, lvq_gaussian_hk=0.2, epochs=1,
                                    lvq_gaussian_thresh=0.005, learning_rate=1, neighborhood='gaussian', verbose=True):

    #muaps = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, muaps)

    # Weight matrices for the SOFM network
    weights = np.zeros((muap_size[0], muap_size[1]), dtype=np.float64)
    weights += init_weight  # Initialize weights
    weights[int(muap_size[0]/2)+1, :] = init_mid_weight_const * muaps[0]  # Modify weight of middle neuron for bias
    if verbose:
        print('Initiated Weight: ' + str(weights.shape))
    node_winning_amount = [ 0 for _ in range(muap_size[0]) ]  # Number of times a neuron wins
    epochs =1  # Training epochs

    # --------------------------Learning Phase------------------------------------------
    # For each epoch
    for _ in range(epochs):
        # For each candidate MUAP waveform
        for i in range(len(muaps)):
            if verbose:
                print('....Current MUAP shape: ' + str(len(muaps[i])))
                print('....Current Weight Shape: ' + str(len(weights[0])))
            # Calculate distance between each output neuron's weight vector and input MUAP waveform
            distance_k = [ np.sum(np.square(np.asarray(muaps[i])-weights[j])) for j in range(muap_size[0])]
            winner_node = np.argmin(distance_k)  # Neuron with minimum distance from the waveform is selected as winner
            node_winning_amount[winner_node] += 1  # Increase the number of times won for the neuron
            if verbose:
                print('....Min Distance for MUAP No. ' + str(i) + ': ' + str(distance_k[winner_node]))

            # Weight adjustment for each output node - LEARNING PHASE 1

            t = i+1  # No.of iteration: starts from 1 and decreases with number of classified MUAP waveforms

            # For each output neuron
            for k in range(len(weights)):
                if verbose:
                    print('........Adjusting weights for outputput node ' + str(k))

                # If the neuron is winner node and it has won exactly one time or the first time
                if (k == winner_node and node_winning_amount[k]==1):
                    gaussian_hk = 1  # Set gaussian threshold to 1
                # Else if the neuron has not won any time
                elif node_winning_amount[k]== 0:
                    gaussian_hk = 1  # Set Gaussian threshold to 1
                # Else
                else:
                    # Set Gaussian theshold based on the number of times the output neuron has won and the current
                    # winner node
                    gaussian_hk = g * math.exp(-(k - winner_node) ** 2 * t / 2) / math.sqrt(node_winning_amount[k])
                if verbose:
                    print('........Gaussian Value: ' + str(gaussian_hk))
                # If the gaussian threshold is below a limit
                if gaussian_hk >= 0.005:
                    if verbose:
                        print('........Adapting Weights')
                    # For each weight in weight vector of the output neuron
                    for x in range(1, weights.shape[1]):
                        # Adjust the weight
                        weights[k][x] = weights[k][x-1] + gaussian_hk * (muaps[i][x] - weights[k][x-1])
            if verbose:
                print('-----------------------------------------------------------------\n')
    node_winning_amount = [0 for _ in range(muap_size[0])]  # Reset the number of times each neuron won

    # Learning Vector Quantization(LVQ) - LEARNING PHASE 2
    if lvq2:
        # For each epoch
        for _ in range(epochs):
            # For each candidate MUAP waveform
            for i in range(len(muaps)):
                if verbose:
                    print('....Current MUAP shape: ' + str(len(muaps[i])))
                    print('....Current Weight Shape: ' + str(len(weights[0])))
                # Calculate distance between each output neuron's weight vector and the MUAP waveform
                distance_k = [ np.sum(np.square(np.asarray(muaps[i])-weights[j])) for j in range(muap_size[0])]
                # The node with minimum distance from the waveform is selected as winner
                winner_node = np.argmin(distance_k)
                # The node with second minimum distance is the second winner node
                second_winner_node = np.argsort(distance_k)[1]
                # Increase the number of times the nodes won
                node_winning_amount[winner_node] += 1
                node_winning_amount[second_winner_node] += 1
                if verbose:
                    print('....Min Distance for MUAP No. ' + str(i) + ': ' + str(distance_k[winner_node]))

                # Weight adjustment for each output node - LEARNING PHASE 1
                g = 1  # 0 < g < 1
                t = i+1  # No.of iteration: starts from 1
                if verbose:
                    print('........Adjusting weights for outputput node ' + str(k))
                    print('........Gaussian Value: ' + str(lvq_gaussian_hk))

                # If the LVQ2 constant is greater than or equals to the LVQ Gaussian threshold
                if lvq_gaussian_hk >= lvq_gaussian_thresh:
                    # For each weight in weight vector of the winner and second winner node
                    for x in range(1, weights.shape[1]):
                        # Adjust weight of the inner node
                        weights[winner_node][x] = weights[winner_node][x - 1] + lvq_gaussian_hk * (muaps[i][x] - weights[winner_node][x - 1])
                        # Adjust weight of the second winner node
                        weights[second_winner_node][x] = weights[second_winner_node][x - 1] \
                                                         - 0.1 * (distance_k[winner_node]/distance_k[second_winner_node]) \
                                                         * lvq_gaussian_hk * (muaps[i][x] - weights[second_winner_node][x - 1])
                # Adjust LVQ2 Threshold for adaptive learning
                lvq_gaussian_hk = 0.2 - 0.01 * node_winning_amount[winner_node]

                # If the LVQ2 Gaussina constant is less than 0
                if lvq_gaussian_hk <0:
                    lvq_gaussian_hk = 0  # Reset the learning constant to 0


    # Classification of actual MUAP and superimposed MUAP
    muap_classification_output = [-1 for _ in range(len(muaps))]  # 0 for actual muap and 1 for superimposed muap
    muap_classification_class = [-1 for _ in range(len(muaps))]  # Classification class for MUAP
    muap_classification_firing_time = [[] for _ in range(len(muaps))]  # Firing time for the classified MUAP waveforms

    # For each candidate MUAP waveform
    for i in range(len(muaps)):
        # Calculate the distance between weight vector of each neuron and the MUAP waveform
        distance_k = [np.sum(np.square(np.asarray(muaps[i]) - weights[j])) for j in range(muap_size[0])]
        # Calculate the winner node based on minimum distance
        winner_node = np.argmin(distance_k)
        # Classify the MUAP waveform as belonging to the class of the winner neuron or motor unit
        muap_classification_class[i] = winner_node
        # Update MUAP firing table
        #muap_firing_table[winner_node].append(muap_firing_time[i])
        # Update Firing time of the classified MUAP waveform
        muap_classification_firing_time[i].append(muap_firing_time[i])

        length_kw = np.sum(weights[winner_node]**2)
        if verbose:
            print('Classification Threshold for MUAP No. ' + str(i+1) + ': ' + str(distance_k[winner_node]/length_kw))

        # If the dissimilarity between winner node and muap waveform is less than 0.2
        if distance_k[winner_node]/length_kw < 0.2:
            muap_classification_output[i] = 0  # Assign the MUAP waveform to the class of winner node
            # Update MUAP firing table
            muap_firing_table[winner_node].append(muap_firing_time[i])
        else:
            muap_classification_output[i] = 1  # Assign the MUAP waveform as superimposed waveform
    if verbose:
        print('Detected Actual MUAP: ' + str(len(muap_classification_output) - np.count_nonzero(muap_classification_output)))
        print('Superimposed MUAP: ' + str(np.count_nonzero(muap_classification_output)))

    # Averaging of MUAP classes containing more than 'n' numbers of members
    if verbose:
        print('INITIAL FIRING TABLE: TOTAL MUAPS: ' + str(len(muaps)))
    # Total current firing in each output neuron/node
    ft = [len(muap_firing_table[i]) for i in range(len(muap_firing_table))]
    if verbose:
        print(ft)
        print('Total Firings: ' + str(np.sum(ft)))
    # Total Actual(0) and Superimposed(1) waveforms in current output node/neuron
    occurences = {i: {0: 0, 1: 0} for i in range(muap_size[0])}
    final_muap_classes = []  # Final Output Node/Neuron Class for each MUAP waveform
    final_muap_outputs = []  # Final Output Type(Actual/Superimposed) Class for each MUAP waveform
    final_muap_waveforms = [] # Final Output MUAP waveforms
    final_muap_firing_times = []  # Final Firing times of the MUAP Waveforms

    # Stores whether each final waveform is averaged result from a MUAP class
    avgd_muap = [False for _ in range(len(muaps))]
    # For each current MUAP waveform
    for i in range(len(muaps)):
        # Update the number of firing/occurence of its consecutive output node
        occurences[muap_classification_class[i]][muap_classification_output[i]] += 1

    # For occurences of each output node
    for c in occurences:
        # If the output node contains more than 3 waveforms
        if occurences[c][0] > 3:
            # Average the waveforms and merge their firing times creating a new waveform
            avg = np.asarray([0 for _ in range(muap_size[1])])  # Averaged Waveform
            avgd_index = []  # Indices of current waveform that are averaged
            firing_time = []  # New merged firing time of the averaged waveform
            # For each current MUAP waveform
            for i in range(len(muaps)):
                # If the waveform belongs to the node whose waveforms are to be averaged and if it is an actual waveform
                if muap_classification_class[i] == c and muap_classification_output == 0:
                    avgd_muap[i] = True  # Set the current MUAP waveform as a member of averaged waveform
                    avgd_index.append(i)  # Add the waveform index which is added to the averaged waveform list for the node
                    avg += np.asarray(muaps[i])  # Add the amplitude of the waveform to the averaged waveform
                    firing_time.append(muap_firing_time[i])  # Add firing time of the waveform to the averaged waveform
            avg = avg / occurences[c][0]  # Average the summed waveform
            # Check whether the new averaged waveform belongs to a neuron class and update the class accordingly
            # else add it to the list of superimposed waveforms

            # Distance of each node's weight vector from the new averaged waveform
            distance_k = [np.sum(np.square(avg - weights[j])) for j in range(muap_size[0])]
            # Node/Neuron with minimum distance is the winner
            winner_node = np.argmin(distance_k)
            # Calculate the dissimilarity between the winner node and new averaged waveform
            length_kw = np.sum(weights[winner_node] ** 2)
            # If the dissimilarity is less than 0.2
            if distance_k[winner_node] / length_kw < 0.2:
                # Add the new averaged waveform to the list of final MUAP waveforms
                final_muap_waveforms.append(avg)
                # Add the class to the list of final classes of MUAP waveforms
                final_muap_classes.append(winner_node)
                # Add the waveform type(Actual) to the list of final output of MUAP waveforms
                final_muap_outputs.append(0)
                # Add firing time of the MUAP waveform to list of final firing times of the MUAP waveform
                final_muap_firing_times.append(firing_time)
                # For each current firing time of the new averaged waveform
                for x in range(len(firing_time)):
                    # Remove the firing time from the firing table of original output node
                    if firing_time[x] in muap_firing_table[c]:
                        muap_firing_table[c].remove(firing_time[x])
                    # Add the firing time to the firing table of new output node
                    muap_firing_table[winner_node].append(firing_time[x])
            else:
                # For each current MUAP waveform that was averaged
                for index in avgd_index:
                    # Reset the averaged status of the current MUAP waveform
                    avgd_muap[index] = False
                    # Add the waveform as superimposed waveform
                    muap_classification_output[index] = 1
    # For each current MUAP waveform that was not averaged
    for i in range(len(muaps)):
        if not avgd_muap[i]:
            # Add it to the list of final MUAP waveforms
            final_muap_waveforms.append(muaps[i])
            # Add its class to the list of class of final MUAP waveforms
            final_muap_classes.append(muap_classification_class[i])
            # Add its output(Actual/Superimposed) to the list of output of final MUAP waveforms
            final_muap_outputs.append(muap_classification_output[i])
            # Add its firing time to the list of final firing times of the MUAP waveforms
            final_muap_firing_times.append(muap_classification_firing_time[i])

    if verbose:
        print('UPDATED FIRING TABLE: TOTAL MUAPS: ' + str(len(muaps)))
        print([len(muap_firing_table[i]) for i in range(len(muap_firing_table))])
    return final_muap_waveforms, final_muap_classes, final_muap_outputs, final_muap_firing_times, muap_firing_table

# Decompose Superimposed MUAP Waveforms from Actual MUAP waveforms and update the firing time
def perform_emg_decomposition(waveforms, waveform_classes, waveform_superimposition, firing_time, max_residue_amp=30,
                        threshold_const_a=0.5, threshold_const_b=4, nd_thresh1=0.2, nd_thresh2=0.5,
                        calculate_endpoints=False, pearson_correlate=True, plot=False, verbose=True):
    if verbose:
        print('Waveform: ' + str(len(waveforms)))
    # Separate Actual and Superimposed Waveform
    actual_muaps = []
    superimposed_muaps = []
    residue_superimposed_muaps = []
    for i in range(len(waveforms)):

        if waveform_superimposition[i] == 0:
            actual_muaps.append([waveforms[i], waveform_classes[i], firing_time[i], firing_time[i][0]])

        else:
            superimposed_muaps.append([waveforms[i], waveform_classes[i], firing_time[i], firing_time[i][0]])
    if verbose:
        print('Actual MUAPS: ' + str(len(actual_muaps)))
        print('Superimposed MUAPS: ' + str(len(superimposed_muaps)))




    # Create a queue that will hold the superimposed waveforms that needs to be decomposed
    superimposed_queue = queue.Queue()
    for i in range(len(superimposed_muaps)):
        superimposed_queue.put(list(superimposed_muaps[i]).copy())
    # For each superimposed waveform perform the following tasks
    cur = 0
    while not superimposed_queue.empty():
        try:
            if verbose:
                print('Superimposed waveform left: ' + str(superimposed_queue.qsize()))
            cur += 1
            smuap = superimposed_queue.get() # Get the superimposed waveform
            # If the Superimposed waveform length is less than that of actual waveform
            if len(smuap[0] < len(actual_muaps[0][0])):
                # Interpolate the end of superimposed waveform
                smuap[0] = list(smuap[0]) + [smuap[0][-1]] * (len(actual_muaps[0][0]) - len(smuap[0]))
            # Cross correlate each reduced MUAP with the superimposed waveform x and find the best matching point
            # i.e. the points where crosscorrelation takes the maximum value

            if verbose:
                print('Crosscorrelating superimposed waveform of length: ' + str(len(smuap[0])))

            best_matching_points = []  # Best matching point and maximum correlation coefficient for each MUAP with the superimposed waveform
            nds = []  # Normalized Euclidean distance for each matching pair
            ads = []  # Average Area Difference for each matching pair
            ths = []  # Varying Threshold for each matching pair
            adjusted_waveforms = []
            # For each actual MUAP waveform
            for j in range(len(actual_muaps)):
                if verbose:
                    print("Correlating with reduced MUAP of length: " + str(len(actual_muaps[j][0])))
                # Calculate the Correlation Vectors from Superimposed MUAP waveform and the actual MUAP waveform
                if pearson_correlate:
                    x = np.asarray(smuap[0]).astype(np.float64) / np.std(smuap[0])
                    y = np.asarray(actual_muaps[j][0]).astype(np.float64) / np.std(actual_muaps[j][0])
                else:
                    x = np.asarray(smuap[0]).astype(np.float64)
                    y = np.asarray(actual_muaps[j][0]).astype(np.float64)
                correlation = correlate(x, y)  # Cross Correlate Superimposed and actual MUAP
                if verbose:
                    print("Cross correlation shape: " + str(len(correlation)))
                    print('Maximum Coefficient: ' + str(np.amax(correlation)) + " at index " + str(
                        np.argmax(correlation)))
                highest_cor_ind = np.argmax(correlation)  # The index with highest correlation value of the two signals
                best_matching_points.append([correlation[highest_cor_ind], highest_cor_ind])
                # Calculate normalized Euclidean Distance, Average Area Difference and Varying Threshold for the
                # matching pair i.e the actual muap and portion of the superimposed waveform that has the highest
                # similarity with the muap

                if highest_cor_ind < len(smuap[0]) - 1:
                    # If the index with highest correlation is less than the length of superimposed waveform
                    if verbose:
                        print('Less')
                    smuap_start = 0  # Set starting index for the cropped superimposed waveform
                    smuap_end = highest_cor_ind + 1 # Set ending index for the cropped superimposed waveform
                    # Set starting index for the cropped actual waveform
                    muap_start = len(actual_muaps[j][0]) - highest_cor_ind - 1
                    # Set ending index for the cropped actual waveform
                    muap_end = len(actual_muaps[j][0])

                elif highest_cor_ind == len(smuap[0]) - 1:
                    smuap_start = 0  # Set starting index for the cropped superimposed waveform
                    smuap_end = len(smuap[0])  # Set ending index for the cropped superimposed waveform
                    muap_start = 0  # Set starting index for the cropped actual waveform
                    muap_end = len(actual_muaps[j][0])  # Set ending index for the cropped actual waveform
                else:
                    if verbose:
                        print('More')
                    # Set starting index for the cropped superimposed waveform
                    smuap_start = highest_cor_ind - (len(smuap[0]) - 1)
                    # Set ending index for the cropped superimposed waveform
                    smuap_end = len(smuap[0])
                    # Set starting index for the cropped actual waveform
                    muap_start = 0
                    # Set ending index for the cropped actual waveform
                    muap_end = len(actual_muaps[j][0]) - (highest_cor_ind - len(smuap[0]) + 1)
                if verbose:
                    print(str(cur) + ', ' + str(j))

                # Get the portion of Superimposed MUAP with highest correlation
                adjusted_superimposed = np.asarray(smuap[0])[smuap_start:smuap_end]
                # Get the portion of Actual MUAP with highest correlation
                adjusted_muap = np.asarray(actual_muaps[j][0])[muap_start:muap_end]
                # Add the adjusted superimposed and actual muaps to the list of adjusted and each actual muap correlation
                adjusted_waveforms.append(
                    [adjusted_superimposed, adjusted_muap, [smuap_start, smuap_end], [muap_start, muap_end]])
                # Calculate the Normalized Euclidean distance between the adjusted superimposed and actual waveform
                nd = np.sum(np.subtract(adjusted_superimposed, adjusted_muap) ** 2) / np.sum(
                    np.multiply(adjusted_muap, adjusted_muap))
                nds.append(nd)
                # Average Area Difference
                ad = np.sum(np.abs(np.subtract(adjusted_superimposed, adjusted_muap))) / len(adjusted_muap)
                ads.append(ad)
                # Varying Area Threshold
                th = threshold_const_b + threshold_const_a * np.sum(np.abs(adjusted_muap)) / len(adjusted_muap)
                ths.append(th)

            # Matching pair with minimum classification coefficient

            best_matching_muap = -1  # The Best matching Actual and current Superimposed MUAP waveform pair
            min_coeff_thresh = sys.maxsize  # Minimum Coefficent threshold of the current best matching pair
            # For each best matching pair of the current superimposed and all actual MUAP waveforms
            for j in range(len(best_matching_points)):
                # If the threshold and Area Difference meets the criteria
                if (nds[j] < nd_thresh1 or (ads[j] < ths[j] and nds[j] < nd_thresh2)):
                    # Calculate the Coefficient threshold for the pair
                    class_coeff = (nds[j] * ads[j]) / (ths[j] * len(adjusted_muap))
                    # If the coefficient threshold is minimum
                    if class_coeff < min_coeff_thresh:
                        best_matching_muap = j # Add its index to the current minimum best matching pair
                        min_coeff_thresh = class_coeff # Add its index to the current minimum coeffcient
            # If a best matching pair with minimum coefficient threshold is found
            if best_matching_muap >= 0:
                # A MUAP Class is identified for the superimposed waveform.
                # Decompose the superimposed waveform from the MUAP class.
                class_smuap = list(smuap[0])  # The superimposed waveform to be decomposed
                class_muap = list(actual_muaps[best_matching_muap][0])  # The actual waveform to be decomposed
                residue_signal = []  # The residue signal after subtracting actual from superimposed signal
                highest_cor_ind = best_matching_points[best_matching_muap][1]  # Highest correlation index of the pair
                # Pad MUAP and SMUAP arrays with zero in order to make them equal length and subtract
                if highest_cor_ind < len(smuap[0]):
                    class_smuap = [0] * adjusted_waveforms[best_matching_muap][3][0] + class_smuap
                    class_muap = class_muap + [0] * adjusted_waveforms[best_matching_muap][3][0]
                    class_smuap_start = adjusted_waveforms[best_matching_muap][3][0]
                    class_smuap_end = adjusted_waveforms[best_matching_muap][3][0] + len(smuap[0])
                elif highest_cor_ind > len(smuap[0]):
                    class_muap = [0] * adjusted_waveforms[best_matching_muap][2][0] + class_muap
                    class_smuap = class_smuap + [0] * adjusted_waveforms[best_matching_muap][2][0]
                    class_smuap_start = 0
                    class_smuap_end = len(smuap[0])

                # Calculate residue signal
                residue_signal = np.subtract(class_smuap, class_muap)
                # Update Firing time of the Best Matching MUAP with the firing time of the superimposed signal
                actual_muaps[best_matching_muap][2] += smuap[2]
                # If the maximum amplitude of the residue signal is less than the maximum residue threshold
                if np.amax(residue_signal[class_smuap_start:class_smuap_end]) < max_residue_amp:
                    # If the max amplitude of residue signal is greater than threshold, then feed it back to the
                    # queue for further decomposition
                    # Replace the initial superimposed waveform with the residue waveform for further decomposition
                    smuap[0] = residue_signal[class_smuap_start:class_smuap_end]
                    # Add the residue signal back to the queue for decomposition
                    superimposed_queue.put(smuap)
                else:
                    # Else add it to the list of decomposed residue signal
                    residue_superimposed_muaps.append(smuap)
                if plot:
                    plt.subplot(3, 1, 1)
                    plt.title('Best Matching MUAP: Cross Correlation Index: ' + str(highest_cor_ind))
                    plt.plot(smuap[0], label='SMUAP')
                    plt.plot(np.arange(highest_cor_ind - len(actual_muaps[best_matching_muap][0]), highest_cor_ind),
                             actual_muaps[best_matching_muap][0],
                             label='MUAP')
                    plt.plot(residue_signal[class_smuap_start:class_smuap_end], label='Residue')
                    plt.grid()
                    plt.legend()
                    plt.subplot(3, 1, 2)
                    plt.title(
                        'Nd: ' + str(nds[best_matching_muap]) + ', Ad: ' + str(ads[best_matching_muap]) +
                        ', Th: ' + str(ths[best_matching_muap]) + ', Coeff: ' + str(min_coeff_thresh))
                    plt.plot(adjusted_waveforms[best_matching_muap][0],
                             label='SMUAP: ' + str(
                                 adjusted_waveforms[best_matching_muap][2][1] -
                                 adjusted_waveforms[best_matching_muap][2][
                                     0]))
                    plt.plot(adjusted_waveforms[best_matching_muap][1], label='MUAP: ' + str(
                        adjusted_waveforms[best_matching_muap][3][1] - adjusted_waveforms[best_matching_muap][3][
                            0]))
                    plt.grid()
                    plt.legend()

                    plt.subplot(3, 1, 3)
                    plt.title('Decomposition')
                    plt.plot(class_smuap, label='SMUAP')
                    plt.plot(class_muap, label='MUAP')
                    plt.plot(residue_signal, label='Residue')
                    plt.grid()
                    plt.legend()

                    plt.show()


            else:
                if verbose:
                    print("No Class identified for the superimposed waveform. Removing it from the list")
                residue_superimposed_muaps.append(smuap)
        except:
            if verbose:
                print("Error occured here")

        else:
            if verbose:
                print("No Class identified for the superimposed waveform. Removing it from the list")
            residue_superimposed_muaps.append(smuap)
    if verbose:
        print('Actual MUAPS: ' + str(len(actual_muaps)))
        print('Superimposed MUAPS: ' + str(len(superimposed_muaps)))
    return actual_muaps, residue_superimposed_muaps
dir = "D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\train\\als\\a01_patient\\N2001A01BB02\\"
a = np.load(dir+"muap_waveforms.npy")
b = np.load(dir+"muap_output_classes.npy")
c = np.load(dir+"muap_superimposition_classes.npy")
d = np.load(dir+"muap_firing_time.npy")
a = np.asarray(a)
c = np.asarray(c)
muap = a[c==0]
#calculate_turns(muap, 6)
