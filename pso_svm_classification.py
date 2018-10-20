import os, random, pywt, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



class PSOSVM:
    def __init__(self):
        ...
    def fit(self, X, y, test_size=0.2):
        ...
    def score(self, X, y):
        ...
    def predict(self, X):
        ...
    def predict_proba(self, X):
        ...
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
def calculate_dwt(data, method='haar', thresholding='soft', level=1):

    if  level<=1:
        (ca, cd) = pywt.dwt(data, method)
        cat = pywt.threshold(ca, np.std(ca) / 2, thresholding)
        cdt = pywt.threshold(cd, np.std(cd) / 2, thresholding)
        return cat, cdt
    else:
        decs = pywt.wavedec(data, method, level=level)
        result=[]
        for d in decs:
            result.append(pywt.threshold(d, np.std(d) / 2, thresholding))
        return result

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

# -----Test PSO-SVM

class Particle:
    """
    Particle objects for the PSO swarm
    """
    def __init__(self):
        self.position = []  # Position of PSO search space particle
        self.velocity = []  # Velocity of PSO search space particle
        self.cost = []  # Cost of PSO search space particle
        self.best = {"position": [], "cost": []}  # Best values obtained so far for PSO search space particle
class PSO:
    def __init__(self, cost_function, upper_bounds, lower_bounds, max_iteration=100, swarm_size = 50, inertia_coefficient=1,
                 inertia_damp=0.99, personal_coefficient=2, global_coefficient=2, verbose=True,
                 cost_function_args=(), fitness_minimize=True, kappa=1, phi1=2.05, phi2=2.05, constriction_coefficient=True):
        # Problem definition

        """
        Constriction Coefficient by Clerc and Kennedy:
        Chi(x) = 2*kappa(k)/(|2 - phi(o) - sqrt( sqr(phi) - 4*phi) |)
        where phi(o) = phi-1(o1) + phi-2(o2)
        Generally,
        k = 1
        o1 = 2.05
        o2 = 2.05
        According to Constriction Coefficient:
        inertia coefficient(w) = chi(x)
        personal coefficient(c1) = chi(x) * phi-1(o1)
        global coefficient = chi(x) * phi-2(o2)
        """
        self.kappa = kappa
        self.phi1 = phi1
        self.phi2 = phi2
        self.phi = self.phi1 + self.phi2
        self.chi = 2*self.kappa/(abs(2 - self.phi - np.sqrt(self.phi**2 - (4*self.phi))))
        self.constriction_coefficient = constriction_coefficient

        self.costFunction = cost_function
        self.costFunctionArgs = cost_function_args  # Arguments to pass to cost function
        self.nVar = len(upper_bounds)  # Number of unknown/decision variables
        self.varSize = np.empty((self.nVar))  # Matrix size of decision variables
        self.varMin = lower_bounds  # Lower bound of decision variables
        self.varMax = upper_bounds  # Upper bound of decision variables
        self.maxVelocity = list((np.asarray(self.varMax) - np.asarray(self.varMin)) * 0.2)
        self.minVelocity = list(np.asarray(self.maxVelocity)*-1)
        self.fitness_minimize = fitness_minimize  # Maximize/Minimize cost function

        # Parameters of PSO
        self.maxIt = max_iteration  # Maximum number of iterations
        self.nPop = swarm_size  # Number of population/swarm in the search space
        self.w_damp = inertia_damp  # Damping ratio of Inertia Coefficient

        if constriction_coefficient:
            self.w = self.kappa  # Inertia Coefficient
            self.c1 = self.chi * self.phi1  # Personal acceleration coefficient
            self.c2 = self.chi * self.phi2  # Global acceleration coefficient
        else:
            self.w = inertia_coefficient  # Inertia Coefficient
            self.c1 = personal_coefficient  # Personal acceleration coefficient
            self.c2 = global_coefficient  # Global acceleration coefficient
        self.verbose = verbose
        self.globalBest = {} # Global best cost and position of the swarm
        self.bestCosts = [] # Best cost at every iteration
        self.bestPositions = [] # Best position at every iteration
        self.particles = [] # particles of the swarm


    def initialize(self):
        if self.fitness_minimize:
            self.globalBest = {"cost": float(sys.maxsize), "position": []}
        else:
            self.globalBest = {"cost": float(-sys.maxsize), "position": []}
        self.bestCosts = []
        self.bestPositions = []
        self.particles = []

        for i in range(self.nPop):
            particle = Particle()
            # Generate random position within the given range for the particles
            particle.position = []
            for j in range(len(self.varMin)):
                if (self.varMin[j] >= -1 and self.varMax[j] <= 1) or (type(self.varMin[j]) == float or type(self.varMax[j]) == float):
                    particle.position.append(random.uniform(self.varMin[j], self.varMax[j]))
                else:
                    particle.position.append(random.randint(self.varMin[j], self.varMax[j]))
            particle.position = np.asarray(particle.position)


            # Generate velocities for the particle
            particle.velocity = np.zeros((len(self.varMin)))

            # Evaluation of the particle
            particle.cost = self.costFunction(particle.position, self.costFunctionArgs)

            # Set current best position for the particle
            particle.best["position"] = particle.position
            # Set current best minimum cost for the particle
            particle.best["cost"] = particle.cost

            # Update global best
            if self.fitness_minimize:
                if particle.best["cost"] < self.globalBest["cost"]:
                    self.globalBest["position"] = np.copy(particle.best["position"])
                    self.globalBest["cost"] = particle.best["cost"]
            else:
                if particle.best["cost"] > self.globalBest["cost"]:
                    self.globalBest["position"] = np.copy(particle.best["position"])
                    self.globalBest["cost"] = particle.best["cost"]

            self.particles.append(particle)
            print("--------------------Particle Number " + str(i+1)  +"--------------------")
            print("Current Position: " + str(particle.position))
            print("Current velocity: " + str(particle.velocity))
            print("Current Best : " + str(particle.best) + "\n")
    def run(self):
        self.initialize()
        # Main Loop of PSO
        if self.verbose:
            plt.show(block=False)
            plt.figure(100)
            plt.grid()
        for it in range(self.maxIt):
            # For each iteration of PSO
            for i in range(self.nPop):
                # For each particle in the current iteration

                # Update velocity of the particle
                v = self.w * self.particles[i].velocity  # Velocity Update

                p = self.c1 * np.random.rand(len(self.varMin)) * (self.particles[i].best["position"] - self.particles[i].position) # Personal Best update
                g = self.c2 * np.random.rand(len(self.varMin)) * (self.globalBest["position"] - self.particles[i].position)
                self.particles[i].velocity = v + p + g

                # Apply Velocity upper and lower bounds
                for j in range(len(self.particles[i].velocity)):
                    self.particles[i].velocity[j] = max(self.particles[i].velocity[j], self.minVelocity[j])
                    self.particles[i].velocity[j] = min(self.particles[i].velocity[j], self.maxVelocity[j])

                # Update position of the particle
                self.particles[i].position = self.particles[i].position + self.particles[i].velocity
                # Apply lower and upper bound limit
                for j in range(len(self.particles[i].velocity)):
                    self.particles[i].position[j] = max(self.particles[i].position[j], self.varMin[j])
                    self.particles[i].position[j] = min(self.particles[i].position[j], self.varMax[j])


                # Update cost of the particle for new position
                self.particles[i].cost = self.costFunction(self.particles[i].position, self.costFunctionArgs)

                if self.fitness_minimize:
                    if self.particles[i].cost < self.particles[i].best["cost"]:
                        # If the current cost calculated is less than the current best cost of the particle
                        self.particles[i].best["position"] = self.particles[i].position  # Update current best position
                        self.particles[i].best["cost"] = self.particles[i].cost  # Update current best cost

                        # Update global best
                        if self.particles[i].best["cost"] < self.globalBest["cost"]:
                            self.globalBest["position"] = np.copy(self.particles[i].best["position"])
                            self.globalBest["cost"] = self.particles[i].best["cost"]
                else:
                    if self.particles[i].cost > self.particles[i].best["cost"]:
                        # If the current cost calculated is less than the current best cost of the particle
                        self.particles[i].best["position"] = self.particles[i].position  # Update current best position
                        self.particles[i].best["cost"] = self.particles[i].cost  # Update current best cost

                        # Update global best
                        if self.particles[i].best["cost"] > self.globalBest["cost"]:
                            self.globalBest["position"] = np.copy(self.particles[i].best["position"])
                            self.globalBest["cost"] = self.particles[i].best["cost"]

            self.bestCosts.append(self.globalBest["cost"])
            self.bestPositions.append(self.globalBest["position"])
            print("Iteration No. " + str(it + 1) + ": Best Cost = " + str(self.globalBest["cost"]) + ", Best Position: " + str(self.globalBest["position"]))
            self.w = self.w * self.w_damp

            if self.verbose:
                x_axis = []
                y_axis = []
                for i in range(self.nPop):
                    x_axis.append(self.particles[i].position[0])
                    if len(self.particles[i].position) > 1:
                        y_axis.append(self.particles[i].position[1])
               # print("X axis: " + str(len(x_axis)))
                #if len(self.particles[i].position) > 1:
                 #   print("Y axis: " + str(y_axis))
                plt.clf()
                if len(self.particles[i].position) > 1:
                    plt.plot(x_axis, y_axis, 'x')
                    plt.xlim((self.varMin[0], self.varMax[0]))
                    plt.ylim((self.varMin[1], self.varMax[1]))
                else:
                    plt.plot(x_axis, 'x')
                    plt.ylim((self.varMin[0], self.varMax[0]))
                plt.pause(0.1)

        # Results
        if self.verbose:

            plt.figure(101)
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.title('Change of Best Cost over iterations')
            plt.plot(self.bestCosts)
            plt.grid()
            plt.xlabel("Iteration")
            plt.ylabel("Best Cost")
            plt.subplot(2, 1, 2)
            plt.title('Change of Position over iterations')
            for k in range(len(self.bestPositions[0])):
                plt.plot(np.asarray(self.bestPositions)[:, k], label='Variable ' + str(k+1))
            plt.legend()
            plt.show()

        return self.particles, self.globalBest, self.bestCosts
def sphere(x, args):
    """
    This is a sample cost function which needs to be minimized. The function returns absolute squared summation
    of the input unknown variables as output and PSO works in order to provide the most optimized input variables(x)
    so that the output(cost) is minimum. In this case the minimum output possible is 0.

    :param x: 1D array containing
    :return: Output Cost (float)
    """

    return np.sum(np.square(x))

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
    classifier = SVC(C=x[0], gamma=x[1], random_state=42)
    if len(args) == 2:
        X_train, X_test, y_train, y_test = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])

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
    "Hello there"
    data_base_dir = 'D:\\thesis\\ConvNet\\MyNet\\temp\\dataset\\'
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

    # 2. Preprocess data set
    print("PREPROCESSING DATA SET")
    raw_data_np = []
    filtered_data_np = []
    features_data = []
    sampling_rates = []

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
        wavelets = calculate_dwt(filtered, method=dwt_wavelet, level=dwt_level)

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

    # Perform Classification
    measure_performance = False
    if not measure_performance:
        features_data_np = np.asarray(features_data)

        print('Feature data: ' + str(features_data_np.shape))
        test_data_size = 1/10
        X_train, X_test, y_train, y_test = train_test_split(features_data_np, labels, test_size=test_data_size,
                                                            random_state=42)

        pso = PSO(svc_optimize, [1, 1], [0.1, 0.1], fitness_minimize=False, cost_function_args=(X_train, y_train),
                  verbose=False)
        svc_particles, svc_global_best, svc_best_costs = pso.run()
        pso = PSO(knn_optimize, [20], [2], fitness_minimize=False, cost_function_args=(X_train, y_train),
                  verbose=False)
        knn_particles, knn_global_best, knn_best_costs = pso.run()

        print("SVC Global best: " + str(svc_global_best))
        print("KNN Global best: " + str(knn_global_best))
        print("----------------------\n------------------\n")
        classifiers = {
            "Support Vector Machine": SVC(probability=True),
            "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
            "Random Forest": RandomForestClassifier(n_estimators=10),
            "PSO-SVM": SVC(C=svc_global_best["position"][0], gamma=svc_global_best["position"][0], probability=True),
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
    else:
        number_folds = 10
        features_data_np = np.asarray(features_data[: len(features_data) - (len(features_data) % number_folds)])
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
            knn_pso = PSO(knn_optimize, [20], [2], fitness_minimize=False, cost_function_args=(X_train, y_train),
                      verbose=False, inertia_damp=0.99, swarm_size=12)
            knn_particles, knn_global_best, knn_best_costs = knn_pso.run()

            rfa_bound_trees = [2, 20]

            rfa_bound_max_features = [0.1, 1]
            rfa_pso = PSO(rfa_optimize, [rfa_bound_trees[1], rfa_bound_max_features[1]], [rfa_bound_trees[0], rfa_bound_max_features[0]],
                          fitness_minimize=False, cost_function_args=(X_train, y_train),
                          verbose=False, inertia_damp=0.99, swarm_size=12)

            rfa_particles, rfa_global_best, rfa_best_costs = rfa_pso.run()
            print("KNN Global best: " + str(knn_global_best))
            classifiers = {
                "Support Vector Machine": SVC(probability=True),
                "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
                "Random Forest": RandomForestClassifier(n_estimators=10),
                "PSO-KNN": KNeighborsClassifier(n_neighbors=int(knn_global_best["position"][0])),
                "PSO-RForest": RandomForestClassifier(n_estimators=int(rfa_global_best["position"][0]), max_features=rfa_global_best["position"][1])
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
        for i in range(len(classifiers)):
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

