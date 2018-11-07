import os, sys, time, random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def load_sample_texts(url, current_files, ext="txt"):
    for f in os.listdir(url):
        fp = os.path.join(url, f)
        if os.path.isfile(fp) and "."+str(ext) in f:
            current_files.append(fp)
        elif os.path.isdir(fp):
            current_files = load_sample_texts(fp, current_files, ext=ext)
    return current_files
def text_segmentation_and_labeling(text_data, total_segments, segment_size=100, pad_text=True, sequential=True,
                                   random_range=[32, 128]):
    segmented_text = []
    segmented_text_label = []
    if sequential:
        for i in range(len(text_data)):
            if len(segmented_text) >= total_segments:
                break
            text = text_data[i]
            if len(text) < segment_size and pad_text:
                for j in range(segment_size - len(text)):
                    text = text + chr(random.randint(random_range[0], random_range[1]))
            if len(text) >= segment_size:
                for j in range(0, len(text), segment_size):
                    if len(segmented_text) >= total_segments:
                        break
                    segment = text[j:j+segment_size]
                    print("Creating label for: =" + segment + "=")
                    sentence_end_marker_str = str(input("Enter Sentence end markers"))
                    sentence_end_marker_arr = sentence_end_marker_str.split(" ")
                    sentence_end_marker_int_arr = [int(sentence_end_marker_arr[k]) for k in range(len(sentence_end_marker_arr))]
                    print("Segmented sentence: ")
                    for k in range(len(sentence_end_marker_int_arr)):
                        if k == 0:
                            print(segment[:sentence_end_marker_int_arr[k]])
                        elif k < len(sentence_end_marker_int_arr)-1:
                            print(segment[sentence_end_marker_int_arr[k-1]:sentence_end_marker_int_arr[k]])
                        else:
                            print(segment[sentence_end_marker_int_arr[k]:])



"""
Detection of sentences from a file
1. Data Conversion from Text to Signal:
   Text file --> Raw Text --> text segmentation -->
   Ascii representation/Signal --> Remove unnecessary characters(e.g. newline, tab, etc.)

2. Unnecessary Character list:
Nul(0) - US(31) except NewLine(10), 127 

"""
# Load sample data
"""plot_signal = False
scale_signal = True
max_ascii = 127
labeled_data_dir = "D:\\text_database\\SentenceCorpus\\labeled_articles\\"
unlabeled_data_dir = "D:\\text_database\\SentenceCorpus\\unlabeled_articles\\"
header_data_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\'
header_sample_data = load_sample_texts(header_data_dir, [], ext="hea")
sample_sentence_labeled = load_sample_texts(labeled_data_dir, [], ext='txt')
sample_sentence_unlabeled = load_sample_texts(unlabeled_data_dir, [], ext='txt')
sample_data = header_sample_data + sample_sentence_labeled + sample_sentence_unlabeled
shuffle = True
if shuffle:
    random.shuffle(sample_data)
print('Total data found: ' + str(len(sample_data)))
print("First sample url: " + sample_data[0])

# Convert each data to string
corrected_sample_urls = []
string_data = []
for i in range((len(sample_data))):
    with open(sample_data[i], 'r') as f:
        try:
            data = f.read()
            if len(data) > 0:
                string_data.append(data)
                corrected_sample_urls.append(sample_data[i])
        except:
            print('Could not read data from ' + str(sample_data[i]))

print('Total String data: ' + str(len(string_data)))
print("First sample string: " + string_data[0])

# Segment Input data
segmented_data = []
text_segmentation_and_labeling(string_data, 10, segment_size=20)

# Convert String to Signal
signal_data = []
for i in range(len(string_data)):
    text = string_data[i]
    signal = [ord(text[j]) for j in range(len(text))]
    signal_data.append(signal)

print("Total Signal data: " + str(len(signal_data)))
print("First sample signal length: " + str(len(signal_data[0])))


# Trim unnecessary characters
trimmed_data = []
for i in range(len(signal_data)):
    signal = signal_data[i]
    trimmed_signal = []
    for j in range(len(signal)):
        if 31 < signal[j] < 127 or signal[j] == 10:
            if scale_signal:
                trimmed_signal.append(signal[j]/max_ascii)
            else:
                trimmed_signal.append(signal[j])
    trimmed_data.append(trimmed_signal)
print("Total Trimmed data: " + str(len(trimmed_data)))
print('First trimmed data length: ' + str(len(trimmed_data[0])))


if plot_signal:
    plt.ion()
    plt.show()
    for i in range(len(trimmed_data)):
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.title("Raw: " + str(corrected_sample_urls[i]))
        plt.plot(signal_data[i])
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.title("Trimmed")
        plt.plot(trimmed_data[i])
        plt.grid(True)
        plt.pause(2)"""


# Temporary: Detection of individual basic waveforms in a signal
from scipy.signal import find_peaks
from MyNet.emg_classification_library.signal_analysis_functions import butter_bandpass_filter
from MyNet.emg_classification_library.dataset_functions import get_dataset

# butter_bandpass_filter(data, cutoff_freqs, btype, fs, order=5)

root = 'D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\'
base = root + 'train\\als\\a01_patient\\N2001A01BB02\\'
urls, labels, label_map = get_dataset(root, shuffle=True)

for index in range(len(urls)):
    plt.clf()
    data = np.load(os.path.join(urls[index], 'data.npy'))
    filtered = butter_bandpass_filter(data, [5, 10000], 'band', 23437.5, order=2)
    peak_ranges = {'low': [0, np.mean(filtered[filtered > 0]) - 1],
                   'high': [np.mean(filtered[filtered > 0]), np.amax(filtered)]}

    # Find peak
    print('Finding peak for peak range type: ' + str(range))
    peak, _ = find_peaks(filtered.copy(), height=peak_ranges['high'], distance=234)
    bep_eep = []
    print('Total Peak detected: ' + str(len(peak)))

    # Find BEP AND EEP
    last_traversed = 0
    if len(peak) > 0:
        for i in range(len(peak)):
            print('Tryng to estimate bep and eep for peak at ' + str(peak[i]))
            current_bep = last_traversed
            # Find BEP

            for j in range(last_traversed + 1, peak[i]):

                if abs(filtered[j] - filtered[j - 1]) / (filtered[j] - filtered[j - 1]) != abs(
                        filtered[j + 1] - filtered[j]) / (filtered[j + 1] - filtered[j]):
                    current_bep = j
                last_traversed = j
            # Find EEP
            current_eep = peak[i] + 1
            if i + 1 < len(peak) and peak[i+1] < len(filtered)-1:
                end = peak[i + 1]
            else:
                end = len(filtered)
            end = len(filtered)-1
            for j in range(peak[i] + 1, end):
                last_traversed = j
                if abs(filtered[j] - filtered[j - 1]) / (filtered[j] - filtered[j - 1]) != abs(
                        filtered[j + 1] - filtered[j]) / (filtered[j + 1] - filtered[j]):
                    current_eep = j
                    break
            bep_eep.append([current_bep, current_eep])

    dist_b2p = np.asarray(peak) - np.asarray(bep_eep)[:, 0]
    amp_b2p = abs(filtered[np.asarray(peak)] - np.abs(filtered[np.asarray(bep_eep)[:, 0]]))
    dist_e2p = np.asarray(bep_eep)[:, 1] - np.asarray(peak)
    amp_e2p = abs(filtered[np.asarray(peak)] - np.abs(filtered[np.asarray(bep_eep)[:, 1]]))

    peak_area = [np.sum(filtered[bep_eep[i][0]: bep_eep[i][1]]**2) for i in range(len(peak))]
    print("Distance b2p: " + str(dist_b2p.shape))
    print("Distance e2p: " + str(dist_e2p.shape))
    plt.subplot(2, 1, 1)
    plt.title('Type: ' + str(label_map[labels[index]]).upper() + ", Left: " + str(len(urls)-index))
    plt.plot(filtered)
    plt.plot(peak, filtered[peak], 'gx')
    plt.plot(np.asarray(bep_eep).flatten(), filtered[np.asarray(bep_eep).flatten()], 'rx')
    plt.grid()
    plt.subplot(2, 1, 2, projection='3d')
    im = plt.scatter(amp_b2p/dist_b2p, amp_e2p/dist_e2p,
                     zs=peak_area, s=5, c=peak/len(filtered), cmap='hot')
    plt.gcf().colorbar(im, shrink=0.5, aspect=5)
    plt.grid()
    plt.xlabel('b2p[sqrt(width**2 + height**2)]')
    plt.ylabel('e2p[sqrt(width**2 + height**2)]')
    plt.show()




