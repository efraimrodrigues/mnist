#python3

#Madeline using the logistic rule for training

import numpy as np
import random
import math

import utils

def lmsrule(w, x, y, epochs, learning_rate):
    for i in range(0, epochs):
        for j in range(0, len(x)):
            y_pred = w @ x[j]

            for index, k in enumerate(y_pred):
                y_pred[index] = (1.0)/(1.0 + math.exp(-k))

            err = y[j] - y_pred

            x_norm = x[j]/(x[j] @ x[j])

            delta = err[:,None] @ x_norm[:,None].T

            w = w + learning_rate*delta

    return w

training_images = utils.training_images()
training_labels = utils.training_labels()

test_images = utils.test_images()
test_labels = utils.test_labels()

n_training_samples = 7000
n_tests = math.floor(n_training_samples/6)
n_rounds = 10
epochs = 25
learning_rate = 0.205

sucess_rate_sum = 0

highest_sucess_rate = 0
lowest_sucess_rate = 1

for r in range(0, n_rounds):
    x_training = []
    y_training = []

    x_test_trans = []
    y_test = []

    max_i = random.randint(0, len(training_images) - n_training_samples)

    training_imgs_range = np.random.permutation(range(0 + max_i, n_training_samples + max_i))

    for i in training_imgs_range:
        x_training.append(np.matrix.flatten(training_images[i]/255))
        
        cod = [0] * 10
        cod[training_labels[i]] = 1
        y_training.append(cod)

    max_i = random.randint(0, len(test_images) - n_tests)

    img_test_range = np.random.permutation(range(0 + max_i, n_tests + max_i))

    for i in img_test_range:
        x_test_trans.append(np.matrix.flatten(test_images[i]/255))

        cod = [0] * 10
        cod[test_labels[i]] = 1
        y_test.append(cod)

    #
    w_init = 0.1 * np.random.rand(len(y_training[0]), len(x_training[0]))

    w = lmsrule(w_init, x_training, y_training, epochs, learning_rate)

    sucess_sum = 0

    max_i = random.randint(0, len(test_images) - n_tests)

    tests_range = np.random.permutation(range(0 + max_i, n_tests + max_i))

    for test in tests_range:
        pred = w @ np.matrix.flatten(test_images[test]/255)

        closest = 0
        for i in range(0, len(pred)):
            if pred[i] > pred[closest]:
                closest = i

        label = test_labels[test]

        if closest == label:
            sucess_sum += 1

    sucess_rate = sucess_sum/n_tests

    sucess_rate_sum += sucess_rate

    if sucess_rate < lowest_sucess_rate:
        lowest_sucess_rate = sucess_rate

    if sucess_rate > highest_sucess_rate:
        highest_sucess_rate = sucess_rate

    print("Round: " + str(r) + " Sucess Rate: " + str(sucess_rate))

mean_sucess_rate = sucess_rate_sum/(n_rounds)

print("Mean sucess rate: " + str(mean_sucess_rate*100) + "%")
print("Highest sucess rate: " + str(highest_sucess_rate*100) + "%")
print("Lowest sucess rate: " + str(lowest_sucess_rate*100) + "%")
#import matplotlib.pyplot as plt
#plt.imshow(training_images[2], cmap='gray')
#plt.show()