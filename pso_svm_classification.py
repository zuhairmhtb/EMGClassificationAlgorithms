import os, random, pywt, sys, pdb, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram
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


def rfa_optimize(x, args):
    classifier = RandomForestClassifier(n_estimators=int(x[0]), max_features=x[1])
    if len(args) == 2:
        X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])

def svc_optimize(x, args):

    classifier = SVC(C=x[0], gamma=x[1], kernel=args[2][int(x[2])])
    if len(args) == 3:
        X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.2, shuffle=False)
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)
    elif len(args) == 5:
        classifier.fit(args[0], args[1])
        return classifier.score(args[3], args[4])

def knn_optimize(x, args):
    classifier = KNeighborsClassifier(n_neighbors=int(x[0]))
    if len(args) == 2:
        X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])

# Main Task starts here
if __name__ == "__main__":
    simulated = False
    if not simulated:
        data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\'
        signal_type = 'Real'
    else:
        data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\simulated_dataset\\'
        signal_type = 'Simulated'
    result_base_dir = "D:\\thesis\\ConvNet\\MyNet\\emg_classification_library\\pso_svm_classification_output\\"
    muscle_location = "Biceps Brachii"
    scale = False

    if scale:
        signal_type += "(Scaled)"

    save_result = False
    #pso = PSO(sphere, [10, 1], [-10, -1], fitness_minimize=True, cost_function_args=(),
     #         verbose=True)
    #pso.run()
    # 1. Get data set
    print("LOADING DATA SET")
    urls, labels, label_map = get_dataset(data_base_dir)
    #urls = urls[:30]
    #labels = labels[:30]
    data_filename = 'data.npy'
    header_filename = 'data.hea'
    print('Dataset Loaded - Total: ' + str(len(urls)) + ', Output Classes: ' + str(len(label_map)))
    print('Label Map: ' + str(label_map))
    print('Labels:  '+ str(type(labels)))
    print('URLS: ' + str(type(urls)))

    save_feature_data = True
    suffix = signal_type + "_" + muscle_location + "_"
    if scale:
        suffix = suffix + "scaled"
    else:
        suffix = suffix + "unscaled"
    feature_data_path = result_base_dir + "features_" + suffix + ".npy"
    raw_data_path = result_base_dir + "raw_" + suffix + ".npy"
    filtered_data_path = result_base_dir + "filtered_" + suffix + ".npy"
    sampling_rate_data_path = result_base_dir + "sampling_rates_" + suffix + ".npy"


    # 2. Preprocess data set
    print("PREPROCESSING DATA SET")
    raw_data_np = []
    filtered_data_np = []
    features_data = []
    sampling_rates = []

    if not os.path.exists(feature_data_path):
        plot_data = False
        for i in range(len(urls)):
            # Load data as numpy array
            d = np.load(os.path.join(urls[i], data_filename))
            fs = read_sampling_rate(os.path.join(urls[i], header_filename))
            label = labels[i]
            raw_data_np.append(d)
            sampling_rates.append(fs)

            # Filter data
            cutoff_high = 10000 # 10 KHz
            cutoff_low = 5 # 5 Hz
            filter_order = 2
            cropped_data_duration = 5000 # ms
            cropped_data = crop_data(d, fs, cropped_data_duration)
            print('Cropped data duration: ' + str(1000*len(cropped_data)/fs))
            filtered = butter_bandpass_filter(cropped_data, [cutoff_low, cutoff_high], 'band', fs, order=filter_order)
            filtered_data_np.append(filtered)
            fs = cutoff_high


            # Calculate DWT
            dwt_wavelet = 'db4'
            dwt_level = 5

            wavelets = calculate_dwt(filtered, method=dwt_wavelet, level=dwt_level, threshold=False)

            #print('Wavelets found: ' + str(len(wavelets)))

            # Calculate Features from dwt
            mean_abs_val = calculate_mav(wavelets)
            #print('Total MAV: ' + str(len(mean_abs_val)))
            avg_pow = calculate_avp(wavelets)
            #print('Total AVP: ' + str(len(avg_pow)))
            std_dev = calculate_std(wavelets)
            #print('Total STD: ' + str(len(std_dev)))
            ratio_abs_mean = calculate_ram(wavelets)
            #print('Total RAM: ' + str(len(ratio_abs_mean)))
            features = []
            for j in range(len(wavelets)):
                features.append(mean_abs_val[j])
                features.append(avg_pow[j])
                features.append(std_dev[j])
                if j < len(wavelets)-1:
                    features.append(ratio_abs_mean[j])
            features_data.append(features)
            #print('Total Features extracted: ' + str(len(mean_abs_val) + len(avg_pow) + len(std_dev) + len(ratio_abs_mean)))


            # Plot data
            if plot_data:
                plt.figure(1)
                plt.clf()
                plt.suptitle('EMG Signal: Subject Type: ' + label_map[label])
                plt.subplot(2, 2, 1)
                plt.title('Raw Signal')
                plt.plot(d)
                plt.grid(True)
                plt.subplots_adjust(hspace=0.5, wspace=0.5)

                plt.subplot(2, 2, 2)
                plt.title('Power Spectral Density(Raw Signal)')
                f, t, sxx = spectrogram(d, fs, nperseg=256, return_onesided=True, mode='psd', scaling='spectrum')
                plt.pcolormesh(t, f, sxx)
                plt.grid(True)
                plt.subplots_adjust(hspace=0.5, wspace=0.5)

                plt.subplot(2, 2, 3)
                plt.title('Filtered Signal: Bandpass - ' + str(cutoff_low) + ', ' + str(cutoff_high))
                plt.plot(filtered)
                plt.grid(True)
                plt.subplots_adjust(hspace=0.5, wspace=0.5)

                plt.subplot(2, 2, 4)
                plt.title('Power Spectral Density(Filtered Signal)')
                f, t, sxx = spectrogram(filtered, fs, nperseg=256, return_onesided=True, mode='psd', scaling='spectrum')
                plt.pcolormesh(t, f, sxx)
                plt.grid(True)
                plt.subplots_adjust(hspace=0.5, wspace=0.5)

            if plot_data:
                plt.figure(2)
                plt.clf()
                plt.suptitle('Discrete Wavelet Transform')
                t_wavelets = len(wavelets)
                for j in range(len(wavelets)):
                    plt.subplot(t_wavelets, 2, j+1)
                    plt.title('Wavelet No. ' + str(j+1))
                    plt.plot(wavelets[j])
                    plt.grid(True)
                    plt.subplots_adjust(hspace=0.5, wspace=0.5)

                    plt.subplot(t_wavelets, 2, j+1+t_wavelets)
                    plt.title('PSD of wavelet no. ' + str(j+1))
                    f, t, sxx = spectrogram(wavelets[i], fs, nperseg=256, return_onesided=True, mode='psd', scaling='spectrum')
                    plt.pcolormesh(t, f, sxx)
                    plt.grid(True)
                    plt.subplots_adjust(hspace=0.5, wspace=0.5)

            if plot_data:
                plt.show()
            print('Left: ' + str(len(urls) - i))

    else:
        features_data = np.load(feature_data_path)
    # Perform Classification
    measure_performance = True
    if not measure_performance:
        result_path = result_base_dir + "psosvm_accuracy_table.html"
        if os.path.exists(result_path):
            with open(result_path, 'r') as fp:
                table = from_html_one(fp.read())
        else:
            table = PrettyTable()
            table.field_names = ["Index No.", "Signal Type", "Muscle Location", "Classifier", "Subject type",
                                 "Train data size", "Test data size", "Subject's test data size", "Accuracy(%)", "Classification Date"]

        features_data_np = np.asarray(features_data)
        if scale:
            features_data_np = preprocessing.scale(features_data_np)

        print('Feature data: ' + str(features_data_np.shape))
        test_data_size = 1/10
        X_train, X_test, y_train, y_test = train_test_split(features_data_np, labels, test_size=test_data_size,
                                                            random_state=42)

        svc_kernels = ['poly', 'rbf', 'sigmoid', 'linear']

        pso = PSO(svc_optimize, [1, 1, len(svc_kernels)], [0.1, 0.1, 0], fitness_minimize=False, cost_function_args=(X_train, y_train, svc_kernels),
                  verbose=True)
        svc_particles, svc_global_best, svc_best_costs = pso.run()

        pso = PSO(svc_optimize, [1, 1, len(svc_kernels)], [0.1, 0.1, 0], fitness_minimize=False,
                  cost_function_args=(X_train, y_train, svc_kernels),
                  verbose=False)
        svc2_particles, svc2_global_best, svc2_best_costs = pso.run()

        pso = PSO(knn_optimize, [20], [2], fitness_minimize=False, cost_function_args=(X_train, y_train),
                  verbose=False)
        knn_particles, knn_global_best, knn_best_costs = pso.run()

        print("SVC Global best: " + str(svc_global_best))
        print("KNN Global best: " + str(knn_global_best))
        print("----------------------\n------------------\n")
        classifiers = {
            "Support Vector Machine(Kernel=Polynomial)": SVC(C=1, kernel='poly', probability=True),
            "Support Vector Machine(Kernel=Radial Basis Function)": SVC(C=1, kernel='rbf', probability=True),
            "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
            "Random Forest": RandomForestClassifier(n_estimators=10),
            "PSO-SVM(Kernel=Polynomial)": SVC(C=svc_global_best["position"][0], gamma=svc_global_best["position"][1],
                           kernel=svc_kernels[int(svc_global_best["position"][2])], probability=True),
            "PSO-SVM(Kernel=RBF)": SVC(C=svc2_global_best["position"][0], gamma=svc2_global_best["position"][1],
                           kernel=svc_kernels[int(svc_global_best["position"][2])], probability=True),
            "PSO-KNN": KNeighborsClassifier(n_neighbors=int(knn_global_best["position"][0]))
        }
        accuracies = []
        preds = []
        probs = []
        for k in classifiers:
            print('------------Classifying with ' + k + "---------------\n")


            classifiers[k].fit(X_train, y_train)
            accuracy = classifiers[k].score(X_test, y_test)
            predictions = list(classifiers[k].predict(X_test))
            probabilites = classifiers[k].predict_proba(X_test)
            preds.append(predictions)
            probs.append(probabilites)
            accuracies.append(accuracy)
            print('Accuracy: ' + str(accuracy))
            print('Target: ' + str(y_test))
            print('Prediction: ' + str(predictions))

            for j in range(len(label_map)):
                correct = 0
                total = 0
                for i in range(len(y_test)):
                    if y_test[i] == j:
                        total += 1
                        if predictions[i] == y_test[i]:
                            correct += 1
                if total > 0:
                    acc = correct/total
                else:
                    acc = -1
                if label_map[j].lower() == "als" or label_map[j].lower() == "neuropathy":
                    l = "Neuropathy"
                elif label_map[j].lower() == "other" or label_map[j].lower() == "normal":
                    l = "Normal/Healthy"
                elif label_map[j].lower() == "myopathy":
                    l = "Myopathy"

                if save_result:
                    #table.field_names = ["Index No.", "Signal Type", "Muscle Location", "Classifier", "Subject type",
                     #                    "Train data size", "Test data size", "Subject's test data size", "Accuracy(%)", "Classification Date"]
                    current_index = len(table._rows) + 1
                    now = datetime.datetime.now()
                    print([current_index, signal_type, muscle_location, k.upper(), l,
                           len(y_train), len(y_test), total, float("{0:.2f}".format(acc * 100)),
                           now.strftime("%Y-%m-%d %H:%M")])
                    table.add_row([current_index, signal_type, muscle_location, k.upper(), l,
                           len(y_train), len(y_test), total, float("{0:.2f}".format(acc * 100)),
                           now.strftime("%Y-%m-%d %H:%M")])

        if save_result and len(table._rows) > 0:
            with open(result_path, 'w') as fp:
                fp.write(table.get_html_string())

    else:
        number_folds = 10
        result_path = result_base_dir + "pso_svm_performance_table.html"
        if os.path.exists(result_path):
            with open(result_path, 'r') as fp:
                table = from_html_one(fp.read())
        else:
            table = PrettyTable()
            table.field_names = ["Index No.", "Signal Type", "Muscle Location", "Classifier", "Subject type",
                                 "Train data size", "Test data size", "Specifity(%)",
                                 "Sensitivity(%)", "Performance accuracy(%)", "Classification Date"]

        features_data_np = np.asarray(features_data[: len(features_data) - (len(features_data) % number_folds)])
        if scale:
            features_data_np = preprocessing.scale(features_data_np)

        inp_labels = list(labels[:len(features_data) - (len(features_data) % number_folds)])
        features_split = [ [] for _ in range(number_folds)]
        labels_split = [ [] for _ in range(number_folds)]
        data_per_fold = int(len(features_data_np)/number_folds)
        class_ratios = []
        nclasses_per_fold = []
        for i in range(len(label_map)):
            class_ratios.append(inp_labels.count(i)/len(inp_labels))
            nclasses_per_fold.append(int( (inp_labels.count(i)/len(inp_labels))*data_per_fold))
        if np.sum(nclasses_per_fold) < data_per_fold:
            extra = data_per_fold - np.sum(nclasses_per_fold)
            while extra > 0:
                nclasses_per_fold[np.argmin(nclasses_per_fold)] += 1
                extra -= 1
        print('Total data: ' + str(len(inp_labels)))
        print('Classes:' + str(label_map))
        print('Class ratio: ' + str(class_ratios))
        print('No. of data per class per fold: ' + str(nclasses_per_fold))
        for i in range(number_folds):
            for j in range(len(nclasses_per_fold)):
                if j in inp_labels:
                    for k in range(nclasses_per_fold[j]):
                        if j in inp_labels:
                            index = inp_labels.index(j)
                            feature = features_data_np[index, :]
                            lab = inp_labels[index]
                            del inp_labels[index]
                            features_data_np = np.delete(features_data_np, index, 0)
                            features_split[i].append(feature)
                            labels_split[i].append(lab)
                        else:
                            break

        for i in range(len(features_split)):
            c = list(zip(features_split[i], labels_split[i]))
            random.shuffle(c)
            features_split[i], labels_split[i] = zip(*c)
            labels_split[i] = list(labels_split[i])
            features_split[i] = list(features_split[i])
        fold_accuracies = []
        fold_predictions = []
        fold_probs = []
        for i in range(len(features_split)):
            X_test = features_split[i]
            y_test = labels_split[i]
            X_train = []
            y_train = []
            for j in range(1, len(features_split)):
                if i != j:
                    X_train = X_train + features_split[j]
                    y_train = y_train + labels_split[j]
            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            print('Train data shape: ' + str(X_train.shape))
            print('Train data labels: ' + str(y_train))
            print('Test data shape: ' + str(X_test.shape))
            print('Test data labels: ' + str(y_test))


            #def __init__(self, cost_function, upper_bounds, lower_bounds, max_iteration=100, swarm_size = 50, inertia_coefficient=1,
                 #inertia_damp=0.99, personal_coefficient=2, global_coefficient=2, verbose=True, cost_function_args=(), fitness_minimize=True)
            #knn_pso = PSO(knn_optimize, [20], [2], fitness_minimize=False, cost_function_args=(X_train, y_train),
             #         verbose=False, inertia_damp=0.99, swarm_size=12)
            #knn_particles, knn_global_best, knn_best_costs = knn_pso.run()

            #rfa_bound_trees = [2, 20]

            #rfa_bound_max_features = [0.1, 1]
            #rfa_pso = PSO(rfa_optimize, [rfa_bound_trees[1], rfa_bound_max_features[1]], [rfa_bound_trees[0], rfa_bound_max_features[0]],
             #             fitness_minimize=False, cost_function_args=(X_train, y_train),
              #            verbose=False, inertia_damp=0.99, swarm_size=12)

            #rfa_particles, rfa_global_best, rfa_best_costs = rfa_pso.run()

            svc_kernels = ['poly', 'rbf', 'sigmoid', 'linear']

            pso = PSO(svc_optimize, [1, 1, len(svc_kernels)-1], [0.1, 0.1, 0], fitness_minimize=False,
                      cost_function_args=(X_train, y_train, svc_kernels),
                      verbose=True, feature_label=["Penalty(C)", "Gamma(G)", "Kernel"], ndview=False)
            svc_particles, svc_global_best, svc_best_costs = pso.run()

            pso = PSO(svc_optimize, [1, 1, len(svc_kernels)-1], [0.1, 0.1, 0], fitness_minimize=False,
                      cost_function_args=(X_train, y_train, svc_kernels),
                      verbose=False)
            svc2_particles, svc2_global_best, svc2_best_costs = pso.run()


            #print("KNN Global best: " + str(knn_global_best))
            classifiers = {
                "Support Vector Machine": SVC(probability=True),
                "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
                "Random Forest": RandomForestClassifier(n_estimators=10),
                #"PSO-KNN": KNeighborsClassifier(n_neighbors=int(knn_global_best["position"][0])),
                #"PSO-RForest": RandomForestClassifier(n_estimators=int(rfa_global_best["position"][0]), max_features=rfa_global_best["position"][1]),
                "PSO-SVM(Kernel=Polynomial)": SVC(C=svc_global_best["position"][0], gamma=svc_global_best["position"][1],
                           kernel=svc_kernels[int(svc_global_best["position"][0])], probability=True),
                "PSO-SVM(Kernel=RBF)": SVC(C=svc2_global_best["position"][0], gamma=svc2_global_best["position"][1],
                               kernel=svc_kernels[int(svc_global_best["position"][0])], probability=True),
            }
            accuracies = []
            preds = []
            probs = []
            for k in classifiers:
                print('------------Classifying with ' + k + "---------------\n")
                classifiers[k].fit(X_train, y_train)
                accuracy = classifiers[k].score(X_test, y_test)
                predictions = list(classifiers[k].predict(X_test))
                probabilites = classifiers[k].predict_proba(X_test)
                preds.append(predictions)
                probs.append(probabilites)
                accuracies.append(accuracy)
                print('Accuracy: ' + str(accuracy))

            fold_accuracies.append(accuracies)
            fold_predictions.append(preds)
            fold_probs.append(probs)
        # Measure Performance of Classifiers

        # 1. Cross Validation Accuracy
        cva = [
            (1/number_folds)* np.sum(np.asarray(fold_accuracies)[:, i]) for i in range(len(classifiers))
        ]
        print("Cross Validation Accuracy: " + str(cva))


        # 2. True Positive, False Positive, True Negative and False Negative
        tp = [[0 for _ in range(len(label_map))] for _ in range(len(classifiers))]
        tn = [[0 for _ in range(len(label_map))] for _ in range(len(classifiers))]
        fp = [[0 for _ in range(len(label_map))] for _ in range(len(classifiers))]
        fn = [[0 for _ in range(len(label_map))] for _ in range(len(classifiers))]
        for i in range(len(label_map)):
            for j in range(len(features_split)):
                for k in range(len(features_split[j])):
                    for z in range(len(classifiers)):
                        if labels_split[j][k] == i and fold_predictions[j][z][k] == i:
                            tp[z][i] += 1
                        elif labels_split[j][k] == i and fold_predictions[j][z][k] != i:
                            fn[z][i] += 1
                        elif labels_split[j][k] != i and fold_predictions[j][z][k] == i:
                            fp[z][i] += 1
                        elif labels_split[j][k] != i and fold_predictions[j][z][k] != i:
                            tn[z][i] += 1
        print('True Positive: ' + str(tp))
        print('True Negative: ' + str(tn))
        print('False Positive: ' + str(fp))
        print('False Negative: ' + str(fn))

        # 3. Calculate sensitivity and specifity
        sensitivity = [[0 for _ in range(len(label_map))] for _ in range(len(classifiers))]
        specifity = [[0 for _ in range(len(label_map))] for _ in range(len(classifiers))]
        performance_acc = [[0 for _ in range(len(label_map))] for _ in range(len(classifiers))]
        i=0
        for k in classifiers:
            for j in range(len(label_map)):
                sensitivity[i][j] = (tp[i][j]/(tp[i][j] + fn[i][j])) * 100
                specifity[i][j] = (tn[i][j]/(tn[i][j] + fp[i][j])) * 100
            for j in range(len(label_map)):
                total_spec = 0
                performance_acc[i][j] = ((sensitivity[i][j] + specifity[i][j])/2)
                print("-------------------------------------------------------------------------")
                print('Performance result for Classifier: "' + list(classifiers.keys())[i] + " and Output Class: " + str(label_map[j]))
                print("Sensitivity: " + str(sensitivity[i][j]))
                print("Specifity: " + str(specifity[i][j]))
                print("Accuracy: " + str(performance_acc[i][j]))
                print("-------------------------------------------------------------------------")


                if label_map[j].lower() == "als" or label_map[j].lower() == "neuropathy":
                    l = "Neuropathy"
                elif label_map[j].lower() == "other" or label_map[j].lower() == "normal":
                    l = "Normal/Healthy"
                elif label_map[j].lower() == "myopathy":
                    l = "Myopathy"
                if save_result:
                    #table.field_names = ["Index No.", "Signal Type", "Muscle Location", "Classifier", "Subject type",
                     #                    "Train data size", "Test data size", "Subject's test data size", "Accuracy(%)", "Classification Date"]
                    current_index = len(table._rows) + 1
                    now = datetime.datetime.now()
                    print([current_index, signal_type, muscle_location, k.upper(), l,
                           len(y_train), len(y_test), specifity[i][j], sensitivity[i][j], performance_acc[i][j],
                           now.strftime("%Y-%m-%d %H:%M")])
                    table.add_row([current_index, signal_type, muscle_location, k.upper(), l,
                           len(y_train), len(y_test), specifity[i][j], sensitivity[i][j], performance_acc[i][j],
                           now.strftime("%Y-%m-%d %H:%M")])

            i += 1
        if save_result and len(table._rows) > 0:
            with open(result_path, 'w') as fp:
                fp.write(table.get_html_string())
