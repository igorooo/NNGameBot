import numpy as np
import math
import random
import operator
from brain import NeuralNetwork


class A(object):
    def __init__(self, a):
        self.a = a

    def __str__(self):
        return str(self.a)


arr = []

arr.append(A(2))
arr.append(A(4))
arr.append(A(1))

print(random.choice(arr))


arr.sort(key=operator.attrgetter('a'), reverse=True)


"""
NN = NeuralNetwork()
input = np.random.rand(6, 1)
print("Theta_1: \n")
print(NN.Theta_1)
print("\n\n\nTheta_2:\n")
print(NN.Theta_2)
print("\n\n\nForward Propagation...\n")
NN.forwardPropagation(input)
print("\nresult: \n")
print(NN.Y)

print(np.shape(NN.Y))

"""
