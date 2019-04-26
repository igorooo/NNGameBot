import numpy as np
import math


def act(x):
    return round(1 / (1 + math.exp(-x)))


def crossFunct(W1, W2):
    return (4*W1 + W2)/5


class NeuralNetwork(object):

    # 0 <= x <= 1
    mutation_probability = 0.1
    bias_mut_prob = 0.1
    theta_mut_prob = 0.05

    def __init__(self):
        self.act = np.vectorize(act)
        self.crossFunct = np.vectorize(crossFunct)

        self.fitness = 0

        self.Theta_1 = np.random.uniform(-2, 2, (3, 6))
        self.Theta_2 = np.random.uniform(-2, 2, (2, 3))
        self.B_1 = np.random.uniform(-2, 2, (3, 1))
        self.B_2 = np.random.uniform(-2, 2, (2, 1))

    def setThetaBias(self, Theta_1, Theta_2, B_1, B_2):
        self.Theta_1 = Theta_1
        self.Theta_2 = Theta_2
        self.B_1 = B_1
        self.B_2 = B_2

    def forwardPropagation(self, X):

        self.A_1_in = (self.Theta_1 @ X) + self.B_1
        self.A_1 = self.act(self.A_1_in)

        self.Y_in = (self.Theta_2 @ self.A_1) + self.B_2
        self.Y = self.act(self.Y_in)

    def crossover(self, NN2):

        childNN = NeuralNetwork()
        child_Theta_1 = self.crossFunct(self.Theta_1, NN2.Theta_1)
        child_Theta_2 = self.crossFunct(self.Theta_2, NN2.Theta_2)
        child_B_1 = self.crossFunct(self.B_1, NN2.B_1)
        child_B_2 = self.crossFunct(self.B_1, NN2.B_1)

        childNN.setThetaBias(child_Theta_1, child_Theta_2, child_B_1, child_B_2)

        return childNN

    def mutation(self):
        np.random.seed(1)

        if(np.random.rand(1) > NeuralNetwork.mutation_probability):
            return False

        for i in range(0, 3):
            if(np.random.rand(1) < NeuralNetwork.bias_mut_prob):
                self.B_1[i, 0] = np.random.uniform(-2, 2, 1)[0]

        for i in range(0, 2):
            if(np.random.rand(1) < NeuralNetwork.bias_mut_prob):
                self.B_2[i, 0] = np.random.uniform(-2, 2, 1)[0]

        for i in range(0, 3):
            for j in range(0, 6):
                if(np.random.rand(1) < NeuralNetwork.theta_mut_prob):
                    self.Theta_1[i, j] = np.random.uniform(-2, 2, 1)[0]

        for i in range(0, 2):
            for j in range(0, 3):
                if(np.random.rand(1) < NeuralNetwork.theta_mut_prob):
                    self.Theta_2[i, j] = np.random.uniform(-2, 2, 1)[0]

        return True
