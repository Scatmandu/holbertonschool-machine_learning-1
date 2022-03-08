#!/usr/bin/env python3

import numpy as np

oh_encode = __import__('24-one_hot_encode').one_hot_encode
oh_decode = __import__('25-one_hot_decode').one_hot_decode

lib = np.load('/Users/isaac/Desktop/School/Holberton/MachineLearning/holbertonschool-machine_learning/supervised_learning/0x01-classification/data/MNIST.npz')
Y = lib['Y_train'][:10]

print(Y)
Y_one_hot = oh_encode(Y, 10)
Y_decoded = oh_decode(Y_one_hot)
print(Y_decoded)
