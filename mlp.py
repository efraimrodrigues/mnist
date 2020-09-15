import numpy as np
import random
import math

import utils as utils

class mlp:
    def __init__(self, x_training, y_training):
        self.x_training = x_training
        self.y_training = y_training
        self.results = []

    def config(self, epochs, eta, mom, n_o_neurons, n_h_layers, n_h_neurons):
        self.epochs = epochs
        self.eta = eta
        self.mom = mom
        self.n_o_neurons = n_o_neurons
        self.n_h_layers = n_h_layers
        self.n_h_neurons = n_h_neurons

        self.hidden_layers = []

        for i in range(0, n_h_layers):
            if i == 0:
                w = 0.01 * np.random.rand(n_h_neurons[i], len(x_training[0]) + 1)
            else:
                w = 0.01 * np.random.rand(n_h_neurons[i], len(self.hidden_layers[i-1]))
            self.hidden_layers.append(w)

        self.output_layer = 0.01 * np.random.rand(len(y_training[0]), n_h_neurons[n_h_layers - 1] + 1)

    def __sigmoide(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def train(self, samples):
        #Store layers' current values 
        old_hidden_layers = []
        for i in range(0, len(self.hidden_layers)):
            old_hidden_layers.append(self.hidden_layers[i])

        old_output_layer = self.output_layer
        for i in range(0, self.epochs):
            quadratic_error = 0

            max_i = random.randint(0, len(self.x_training) - samples)

            training_range = np.random.permutation(range(0 + max_i, samples + max_i))

            for j in training_range:
                #Hidden layer neuron's activation
                u = []
                y = []

                #Calculates hidden layers
                for k in range(0, self.n_h_layers):
                    if k == 0:
                        u_k = self.hidden_layers[k] @ (np.insert(self.x_training[j], 0, 1))
                    else:
                        u_k = self.hidden_layers[k] @ y[k-1] 

                    u.append(u_k)
                    
                    y_k = []
                    for l in range(0, len(u_k)):
                        y_k.append(self.__sigmoide(u_k[l]))

                    y.append(y_k)
                
                hidden_layer_output = y[len(self.hidden_layers) - 1]

                #Computes output layer
                o = self.output_layer @ np.array(np.insert(hidden_layer_output, 0, 1)).T
                for k in range(0, len(o)):
                    o[k] = self.__sigmoide(o[k])

                error = y_training[j] - o
                quadratic_error = quadratic_error + 0.5*(pow(error, 2))

                #Output layer gradient
                output_d = np.multiply(error, np.multiply(o, 1 - o) + 0.05)

                #Hidden layer gradient
                hidden_d = [None] * len(self.hidden_layers)
                for k in range(len(self.hidden_layers) - 1, -1, -1):
                    d = []
                    if k == len(self.hidden_layers) - 1:
                        d = np.multiply(
                                    np.multiply(y[k], 1 - np.array(y[k])) + 0.05,
                                    np.transpose(self.output_layer[:, 1:]) @ output_d)
                    else:
                        d = np.multiply(
                                    np.multiply(y[k], 1 - np.array(y[k])) + 0.05,
                                    np.transpose(self.hidden_layers[k+1]) @ hidden_d[k+1]
                                )

                    hidden_d[k] = d

                aux_output_layer = self.output_layer
                self.output_layer = (self.output_layer
                                    + self.eta
                                    * np.array(output_d[:, None]) @ np.array(np.insert(hidden_layer_output, 0, 1)[:, None]).T
                                    + mom * (self.output_layer - old_output_layer)
                                    )
                old_output_layer = aux_output_layer

                for k in range(0, len(self.hidden_layers)):
                    aux_hidden_layer = self.hidden_layers[k]
                    if k == 0:
                        self.hidden_layers[k] = (self.hidden_layers[k] 
                                                + self.eta
                                                * np.array(hidden_d[k])[:, None] @ np.array(np.insert(self.x_training[j], 0, 1)[:, None]).T
                                                + mom * (self.hidden_layers[k] - old_hidden_layers[k])
                                            )
                    else:
                        self.hidden_layers[k] = (self.hidden_layers[k] 
                                                + self.eta
                                                * np.array(hidden_d[k])[:, None] @ np.array(y[k-1])[:, None].T
                                                + mom * (self.hidden_layers[k] - old_hidden_layers[k])
                                            )
                    old_hidden_layers[k] = aux_hidden_layer

    def test(self, x_test, y_test):
        success_sum = 0
        for i in range(0, len(x_test)):
            #Hidden layer neuron's activation
            u = []
            y = []

            #Calculates hidden layers
            for k in range(0, self.n_h_layers):
                if k == 0:
                    u_k = self.hidden_layers[k] @ (np.insert(x_test[i], 0, 1))
                else:
                    u_k = self.hidden_layers[k] @ y[k-1]
 
                u.append(u_k)
                
                y_k = []
                for l in range(0, len(u_k)):
                    y_k.append(self.__sigmoide(u_k[l]))

                y.append(y_k)

            hidden_layer_output = y[len(self.hidden_layers) - 1]

            #Computes output layer
            output = self.output_layer @ np.array(np.insert(hidden_layer_output, 0, 1)).T
            for k in range(0, len(output)):
                output[k] = self.__sigmoide(output[k])

            closest = 0
            for j in range(0, len(output)):
                if output[j] > output[closest]:
                    closest = j

            label = y_test[i]

            if label[closest] == 1:
                success_sum += 1

        self.results.append(100 * success_sum/len(x_test))

n_training_samples = 7000
n_tests = math.floor(n_training_samples/6)
n_rounds = 5
epochs = 15
learning_rate = 0.065
mom = 0.85

sucess_rate_sum = 0

highest_sucess_rate = 0
lowest_sucess_rate = 1

training = utils.training_set()
tests = utils.testing_set()

x_training = training[0]
y_training = training[1]

net = mlp(x_training, y_training)

for i in range(0, n_rounds):
    #net.config(epochs, learning_rate, mom, len(y_training[0]), 1, 25)
    net.config(epochs, learning_rate, mom, len(y_training[0]), 2, [30, 25])
    net.train(n_training_samples)
    net.test([tests[0][i] for i in range(0, n_tests)], [tests[1][i] for i in range(0, n_tests)])
    print(net.results)

