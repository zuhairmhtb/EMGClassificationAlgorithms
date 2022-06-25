from scipy.signal import butter, lfilter, spectrogram, find_peaks
from sklearn.svm import SVC, SVR
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from particle_swarm_optimization import Particle, PSO
# Optimization Function for KNearest Neighbor Classifier
def knn_optimize(x, args):
    classifier = KNeighborsClassifier(n_neighbors=int(x[0]))
    if len(args) == 2:
        X_tr, X_te, y_tr, y_te = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_tr, y_tr)
        return classifier.score(X_te, y_te)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])
# Create, Optimize and return a K-Nearest Neighbor classifier optimized using PSO
def getKNN(X_input, y_input):
    classifier_neighbor_range = [1, 5]
    pso = PSO(knn_optimize, [classifier_neighbor_range[1]], [classifier_neighbor_range[0]],
              fitness_minimize=False, cost_function_args=(X_input, y_input),
              verbose=False, ndview=False, max_iteration=50)
    knn_particles, knn_global_best, knn_best_costs = pso.run()
    classifier = KNeighborsClassifier(n_neighbors=int(knn_global_best["position"][0]))
    return classifier, (knn_particles, knn_global_best, knn_best_costs)


# Optimization Function for Random Forest Algorithm
def rfa_optimize(x, args):
    classifier = RandomForestClassifier(n_estimators=int(x[0]), warm_start=True)

    if len(args) == 2:
        X_tr, X_te, y_tr, y_te = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_tr, y_tr)
        return classifier.score(X_te, y_te)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])
# Create, Optimize and return a Random Forest classifier optimized using PSO
def getRFA(X_input, y_input):
    decision_tree_range = [5, 15]
    pso = PSO(rfa_optimize, [decision_tree_range[1]], [decision_tree_range[0]],
              fitness_minimize=False, cost_function_args=(X_input, y_input),
              verbose=False, ndview=False, max_iteration=50)
    rfa_particles, rfa_global_best, rfa_best_costs = pso.run()
    classifier = RandomForestClassifier(n_estimators=int(rfa_global_best["position"][0]), warm_start=True)
    return classifier, (rfa_particles, rfa_global_best, rfa_best_costs)


# Optimization Function For SVM with RBF Kernel
def svmrbf_optimize(x, args):
    classifier = SVC(kernel='rbf', C=float(x[0]), gamma=float(x[1]))
    if len(args) == 2:
        X_tr, X_te, y_tr, y_te = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_tr, y_tr)
        return classifier.score(X_te, y_te)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])

# Create, Optimize and return a SVM classifier with RBF Kernel and optimized using PSO
def getSVMRBF(X_input, y_input):
    penalty_param_range = [0.01, 10000]
    gamma_param_range = [0.0001, 10]
    pso = PSO(svmrbf_optimize, [penalty_param_range[1], gamma_param_range[1]], [penalty_param_range[0], gamma_param_range[0]],
              fitness_minimize=False, cost_function_args=(X_input, y_input),
              verbose=False, ndview=False, max_iteration=50)
    svm_particles, svm_global_best, svm_best_costs = pso.run()

    classifier = SVC(kernel='rbf', C=float(svm_global_best["position"][0]), gamma=float(svm_global_best["position"][1]),
                     probability=True)
    return classifier, (svm_particles, svm_global_best, svm_best_costs)


# Optimization Function For SVM with Polynomial Kernel
def svmpol_optimize(x, args):
    classifier = SVC(kernel='poly', C=float(x[0]), gamma=float(x[1]))
    if len(args) == 2:
        X_tr, X_te, y_tr, y_te = train_test_split(args[0], args[1], test_size=0.2, shuffle=True)
        classifier.fit(X_tr, y_tr)
        return classifier.score(X_te, y_te)
    elif len(args) == 4:
        classifier.fit(args[0], args[1])
        return classifier.score(args[2], args[3])

# Create, Optimize and return a SVM classifier with Polynomial Kernel and optimized using PSO
def getSVMPOL(X_input, y_input):
    penalty_param_range = [0.1, 1]
    gamma_param_range = [0.1, 1]
    pso = PSO(svmpol_optimize, [penalty_param_range[1], gamma_param_range[1]], [penalty_param_range[0], gamma_param_range[0]],
              fitness_minimize=False, cost_function_args=(X_input, y_input),
              verbose=False, ndview=False, max_iteration=50)
    svm_particles, svm_global_best, svm_best_costs = pso.run()
    classifier = SVC(kernel='poly', C=float(svm_global_best["position"][0]), gamma=float(svm_global_best["position"][1]),
                     probability=True)
    return classifier, (svm_particles, svm_global_best, svm_best_costs)