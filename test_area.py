import os, sys, time, random
import numpy as np
import matplotlib.pyplot as plt
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
plot_signal = False
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
        plt.pause(2)


