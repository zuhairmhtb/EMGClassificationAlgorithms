import os, random, pywt, sys, pdb, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram
from sklearn.svm import SVC, SVR
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#from prettytable import PrettyTable, from_csv, from_html_one
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
                 inertia_damp=0.99, personal_coefficient=2, global_coefficient=2, verbose=True, ndview=True, feature_label=[],
                 cost_function_args=(), fitness_minimize=True, kappa=1, phi1=2.05, phi2=2.05, constriction_coefficient=True,):
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
        self.enable_ndview = ndview
        self.ndview_xlabel = feature_label

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
                if not self.enable_ndview:
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
                else:

                    plot_data = np.zeros((self.nPop, len(self.varMin)), dtype=np.float64)
                    for i in range(plot_data.shape[0]):
                        for j in range(plot_data.shape[1]):
                            plot_data[i][j] = self.particles[i].position[j]
                    plt.pcolormesh(plot_data, cmap='hot')
                    if len(self.ndview_xlabel) == 0:
                        plt.xticks(np.arange(0, plot_data.shape[1]), np.arange(0, plot_data.shape[1]))
                    else:
                        plt.xticks(np.arange(0, len(self.ndview_xlabel)), self.ndview_xlabel)
                    plt.grid(True)
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


        return self.particles, self.globalBest, self.bestCosts