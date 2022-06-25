This is a python package that contains different algorithm proposed in different research papers in order to perform EMG classification
The Algorithms currently implemented for classification of EMG are:
1. Classification of EMG signals using PSO optimized SVM for diagnosis of neuromuscular disorders (DOI: https://doi.org/10.1016/j.compbiomed.2013.01.020)
2. IDENTIFYING THE MOTOR NEURON DISEASE IN EMG SIGNAL USING TIME AND FREQUENCY DOMAIN FEATURES WITH COMPARISON (DOI: 10.5121/sipij.2012.3207)

Files and Folders in the package:
1. Performance Metrics, Accuracy Graph and Table, Signal Preprocessing Graphs, Feature Array, etc. obtained from each
   classification technique are included in '*_*_classification_output' folder of the base directory.
       e.g. The classification outputs for Algorithm 1 are included in 'pso_svm_classification_output' folder and
       the outputs for Algorithm 2 are included in 'time_freq_classification output folder'.

2. init.py is the default file of the package.

3. 'dataset_functions.py' contains the functions associated in retrieving data urls and labels from specified
   data set directory.

4. 'muap_analysis_functions.py' contains the functions associated with extracting features from Muscle Unit Action
    Potential(MUAP) waveforms generated from EMG Decomposition technique.
    (N.B: The Algorithm for EMG Decomposition will be added later.)

5. 'particle_swarm_optimization.py' contains the functions associated with performing Particle Swarm Optimization(PSO)
    technique in order to obtain best hyperparameters for a classifier or any other optimization task.

6. 'pso_svm_classification.py' is an obsolete file which will be later replaced by 'wavelet_transform_classification.py'.

7. 'signal_analysis_functions.py' contains the functions associated with Preprocessing and extracting features from
   time and frequency domain of a signal.

8. 'test_area.py' is an additional file for testing code under development.

9. 'time_freq_classification.py' contains the functions associated with performing and evaluating classification of EMG
   signals using Algorithm 2.

10. 'utility.py' is an additional file for manipulating data structure and organization e.g. copying dataset to a
    different directory, renaming dataset, etc.

11. 'wavelet_transform_classification.py' contains the functions associated with performing and evaluating classification
    of EMG signals using Algorithm 1.

Dataset Structure:
    1. The main dataset directory consists of two folders (train and test) where the 'test' folder contains patient
       records for testing  the classifier and 'train' folder contains patient record for training the classifier.

    2. Each of the train/test directory consists of three folders(myopathy, ALS and normal) where each folder contains
       folders for different subjects who falls under the specified group. The record folders of each subject is stored in
       a folder(patient folder) bearing a unique ID number(e.g. a01_patient, c01_patient, etc.) for each individual
       patient. Each patient folder can have multiple EMG record folders obtained from the brachial biceps of the
       subject. Each record folder contains information related to each signal recorded from the specific patient.

    3. Each record folder of the subjects also bear a unique ID number(e.g. N2001A01BB05, N2001A01BB06, etc.) and each
       folder contains three files. They are 'data.npy' and 'data.hea'.

    4. data.npy: This file contains the EMG signal recorded from an electromyograph. The data is stored as a 'Numpy' one
       dimensional array where the length of array indicates the number of samples obtained at a specific sampling frequency.

    5. data.hea: This is a WFDB header file that contains all the information regarding subject under investigation and
       recorded EMG signal from the subject. As for example it contains the sampling frequency and total number of
       samples obtained from the signal, gender of the subject, period of diagnosis, duration of disease, location of
       placement of electrode, filters used, level of insertion of needle, etc. The data is stored as a text file
       (More documentation on WFDB header files: http://www.emglab.net/emglab/Tutorials/WFDB.html).

Preprocessing, Feature Extraction and Classification:
    1. Performed according to the techniques mentioned in respective papers of each implemented EMG Classification
       algorithm.

Performance Evaluation:
    1. ROC Curve and Area Under the Curve for varying input size
    2. Test & Validation accuracy, Sensitivity and Specificity Curve for each Feature Extraction technique
    3. Average Performance Table for varying input size and each Feature Extraction Technique
    4. Feature Output Curves and Tables
    5. Signal preprocessing Curves.

Software specifications:

1. Python version: 3.5.2

2. OS: Windows

3. Install python packages using: pip install -r requirements.txt

4. Run python script: python emg_decomposition_classification.py

