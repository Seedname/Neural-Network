import numpy as np
import csv

np.set_printoptions(threshold=np.inf)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 
    # return max(0, x)
    # return 

def sig_deriv(u):
    # u already has sigmoid(x) applied to it
    u = np.array(u)
    return u * (1 - u)
    # return np.abs(u)/(2*u) + 0.5

class Network:
    def __init__ (self, shape):
        self.shape = shape

        self.weights = [np.random.randn(shape[i],shape[i-1]) for i in range(1, len(shape))]
        self.biases =  [np.random.randn(shape[i], 1) for i in range(1, len(shape))]

    def feed_forward (self, input):
        activation = input
        activations = [activation]

        # Feed forward
        for L in range(len(self.weights)):
            weight = self.weights[L]
            bias = self.biases[L]
            
            z = np.dot(weight, activation) + bias

            activation = sigmoid(z)
            activations.append(np.mat(activation))

        return activations

    def backpropagation (self, input, target):

        activations = self.feed_forward(input)
        # print(activations[-1].shape)
        # print(target.shape)
    
        # Find gradient of cost for weights and biases for the entire layer
        cost_gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        cost_gradient_biases  = [np.zeros(bias.shape) for bias in self.biases]

        multiplier = np.mat(np.array(activations[-1] - target.T) * sig_deriv(activations[-1]))
        cost_gradient_weights[-1] = np.dot(multiplier, activations[-2].T)
        cost_gradient_biases[-1]  = multiplier

        for L in range(2, len(self.shape)):
            activation = activations[-L]
            # dz/da = w 
            multiplier = np.mat(np.array(np.dot(self.weights[-L+1].T, multiplier)) * sig_deriv(activation))
            cost_gradient_weights[-L] = np.dot(multiplier, activations[-L-1].T)
            cost_gradient_biases[-L]  = multiplier

        return (cost_gradient_weights, cost_gradient_biases)

    def train (self, inputs, targets, epochs=30, mini_batch_size=64, learning_rate=0.01, patience=5, test_data=None):
        training_data = list(zip(inputs, targets))
        patience_threshold = 2
        prev_accuracy = 0
        patience_counter = 0

        for full_pass in range(epochs):
            print(f'Epoch {full_pass+1}/{epochs}')
            np.random.shuffle(training_data)

            for i in range(0, len(training_data), mini_batch_size):
                # print(f'Mini Batch {i}/{len(training_data)//mini_batch_size}')
                mini_batch = training_data[i:i+mini_batch_size]

                cost_gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
                cost_gradient_biases  = [np.zeros(bias.shape) for bias in self.biases]

                # weight_velocities = [0 for _ in range(len(self.weights))]
                # bias_velocities = [0 for _ in range(len(self.biases))]

                for input, target in mini_batch:
                    cost_gradient_weight, cost_gradient_bias = self.backpropagation(input, target)

                    for j in range(len(self.weights)):
                        cost_gradient_weights[j] += cost_gradient_weight[j]
                        cost_gradient_biases[j]  += cost_gradient_bias[j]

                for j in range(len(self.weights)):
                    # weight_velocities[j] = mass * weight_velocities[j] + learning_rate * cost_gradient_weights[j]
                    # bias_velocities[j] = mass * bias_velocities[j] + learning_rate * cost_gradient_biases[j]
                    
                    # self.weights[j] -= weight_velocities[j]
                    # self.biases[j]  -= bias_velocities[j]

                    # weight_velocities[j] = mass * weight_velocities[j] + cost_gradient_weights[j]
                    # self.weights[j] -= learning_rate * weight_velocities[j]

                    # bias_velocities[j] = mass * bias_velocities[j] + cost_gradient_biases[j]
                    # self.biases[j] -= learning_rate * bias_velocities[j]

                    self.weights[j] -= learning_rate * cost_gradient_weights[j]
                    self.biases[j]  -= learning_rate * cost_gradient_biases[j]

            if test_data:
                correct = 0
                for input, target in test_data:
                    result = neural_network.feed_forward(input)[-1]
                    if np.argmax(result) == np.argmax(target):
                        correct += 1

                accuracy = 100*correct/len(test_data)
                print(f"Accuracy: {accuracy}%")
                # print(abs(accuracy-prev_accuracy))
                if abs(accuracy - prev_accuracy) <= patience_threshold:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter > patience:
                    print("Network not improving fast enough, quit to avoid overfitting")
                    break

                prev_accuracy = accuracy
            
    

        f = open('weights.txt', 'w')
        f.write(str(self.weights))
        f.close()

        f = open('biases.txt', 'w')
        f.write(str(self.biases))
        f.close()


rows = []
with open('mnist_train.csv') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        rows.append(row)

target_values = [np.asmatrix([1 if i == int(row[0]) else 0 for i in range(10)]) for row in rows]
input_values = [np.asmatrix(list(map(lambda x : x / 255, map(int, row[1:])))).T for row in rows]


rows = []
with open('mnist_test.csv') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        rows.append(row)

target_values_test = [np.asmatrix([1 if i == int(row[0]) else 0 for i in range(10)]) for row in rows]
input_values_test = [np.asmatrix(list(map(lambda x : x / 255, map(int, row[1:])))).T for row in rows]

test_data = list(zip(input_values_test, target_values_test))
neural_network = Network([784, 128, 10])
neural_network.train(input_values, target_values, 50, 64, 0.1, test_data=test_data)


# neural_network = Network([2, 3, 2])
# times = 100
# zerozero = [np.asmatrix([0,0]).T for _ in range(times)]
# zeroone =  [np.asmatrix([0,1]).T for _ in range(times)]
# onezero =  [np.asmatrix([1,0]).T for _ in range(times)]
# oneone  =  [np.asmatrix([1,1]).T for _ in range(times)]

# input_values = zerozero + zeroone + onezero + oneone

# zerozero = [np.asmatrix([1,0]) for _ in range(times)]
# zeroone =  [np.asmatrix([0,1]) for _ in range(times)]
# onezero =  [np.asmatrix([0,1]) for _ in range(times)]
# oneone  =  [np.asmatrix([1,0]) for _ in range(times)]

# target_values = zerozero + zeroone + onezero + oneone
# # print(input_values[0].shape)

# neural_network.train(input_values, target_values, 100, 1, 1)

# print(np.argmax(neural_network.feed_forward(np.asmatrix([0,0]).T)[-1]))
# print(np.argmax(neural_network.feed_forward(np.asmatrix([0,1]).T)[-1]))
# print(np.argmax(neural_network.feed_forward(np.asmatrix([1,0]).T)[-1]))
# print(np.argmax(neural_network.feed_forward(np.asmatrix([1,1]).T)[-1]))
