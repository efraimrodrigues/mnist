#Least Squares Estimation Method

import numpy as np
import random 

import utils

training_images,training_labels = utils.training_set()

test_images, test_labels = utils.testing_set()

n_training_samples = 20000
n_tests = 2000
n_rounds = 10

sucess_rate_sum = 0

highest_sucess_rate = 0
lowest_sucess_rate = 1

rounds_sucess_rate = []

#Using only the first 1000 tests led to a sucess rate of something around 82% regardless the number of training samples.
#Increasing the number of tests led to a higher sucess rate of something around 85%. Maybe the first thousand tests are biased or are not enough.
for r in range(0, n_rounds):
    x_trans = []
    y = []

    max_i = random.randint(0, len(training_images) - n_training_samples)

    training_imgs_range = np.random.permutation(range(0 + max_i, n_training_samples + max_i))

    for i in training_imgs_range:
        x_trans.append(np.matrix.flatten(training_images[i]/255))

        y.append(training_labels[i])

    x = np.transpose(x_trans)
    y = np.transpose(y)

    yx = np.dot(y, x_trans)

    xx_inv = np.linalg.pinv(np.dot(x, x_trans))

    w = np.dot(yx, xx_inv)

    sucess_sum = 0

    max_i = random.randint(0, len(test_images) - n_tests)

    tests_range = np.random.permutation(range(0 + max_i, n_tests + max_i))

    for test in tests_range:
        pred = np.dot(w, np.matrix.flatten(test_images[test]/255))

        closest = 0
        for i in range(0, len(pred)):
            if pred[i] > pred[closest]:
                closest = i

        label = test_labels[test].index(1)

        if closest == label:
            sucess_sum += 1

    sucess_rate = sucess_sum/n_tests

    sucess_rate_sum += sucess_rate

    if sucess_rate < lowest_sucess_rate:
        lowest_sucess_rate = sucess_rate

    if sucess_rate > highest_sucess_rate:
        highest_sucess_rate = sucess_rate

    rounds_sucess_rate.append(sucess_rate)

    print("Round: " + str(r) + " Sucess Rate: " + str(sucess_rate))

mean_sucess_rate = sucess_rate_sum/(n_rounds)

print("Mean sucess rate: " + str(mean_sucess_rate*100) + "%")
print("Highest sucess rate: " + str(highest_sucess_rate*100) + "%")
print("Lowest sucess rate: " + str(lowest_sucess_rate*100) + "%")
print("Standard Deviation: " + str(np.std(rounds_sucess_rate)))