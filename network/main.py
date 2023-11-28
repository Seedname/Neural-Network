import numpy as np
import csv
from matplotlib import pyplot as plt 
import multiprocessing as mp
import time

np.random.seed(42)

np.set_printoptions(threshold=np.inf)

def act_func(x):
    return 1 / (1 + np.exp(-x)) 
    
def act_deriv(x):
    return act_func(x) * (1 - act_func(x))

activation_function = np.vectorize(act_func)
activation_function_derivative = np.vectorize(act_deriv)

class Network:
    def __init__ (self, shape):
        self.shape = shape

        self.weights = [np.sqrt(2/(shape[i-1] + shape[i])) * np.random.randn(shape[i],shape[i-1]) for i in range(1, len(shape))]
        self.biases =  [0 * np.random.randn(shape[i], 1) for i in range(1, len(shape))]

    def feed_forward (self, input):
        activation = input
        activations = [activation]
        zs = []

        # Feed forward
        for L in range(len(self.weights)):
            weight = self.weights[L]
            bias = self.biases[L]
            
            z = np.dot(weight, activation) + bias
            zs.append(z)

            activation = activation_function(z)
            activations.append(np.mat(activation))

        return (activations, zs)
    
    def backpropagation (self, input, target):
        activations, zs = self.feed_forward(input)

        # Find gradient of cost for weights and biases for the entire layer
        cost_gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        cost_gradient_biases  = [np.zeros(bias.shape) for bias in self.biases]

        multiplier = np.mat(np.array(activations[-1] - target.T) * activation_function_derivative(np.array(zs[-1])))
        cost_gradient_weights[-1] = np.dot(multiplier, activations[-2].T)
        cost_gradient_biases[-1]  = multiplier

        for L in range(2, len(self.shape)):
            # dz/da = w 
            multiplier = np.mat(np.array(np.dot(self.weights[-L+1].T, multiplier)) * activation_function_derivative(np.array(zs[-L])))
            cost_gradient_weights[-L] = np.dot(multiplier, activations[-L-1].T)
            cost_gradient_biases[-L]  = multiplier

        # queue.put((cost_gradient_weights, cost_gradient_biases))
        return (cost_gradient_weights, cost_gradient_biases)

    def train (self, inputs, targets, epochs=30, mini_batch_size=64, learning_rate=0.01, patience=5, padding=0, test_data=None):
        training_data = list(zip(inputs, targets))
        patience_threshold = 2
        prev_accuracy = 0
        patience_counter = 0
        prev_time = time.time()
        y = []
        y2 = []

        for full_pass in range(epochs):
            print(f'Epoch {full_pass+1}/{epochs}')
            np.random.shuffle(training_data)

            for i in range(0, len(training_data), mini_batch_size):
                mini_batch = training_data[i:i+mini_batch_size]

                cost_gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
                cost_gradient_biases  = [np.zeros(bias.shape) for bias in self.biases]

                # output = mp.Queue()
                # processes = [mp.Process(target=self.backpropagation, args=(output, input, target)) for input, target in mini_batch]

                # # Run processes
                # for p in processes:
                #     p.start()
                #     p.join()

                # # Exit the completed processes
                # for p in processes:
                    

                # Get process results from the output queue
                # results = [output.get() for p in processes]

                # for result in results:
                #     cost_gradient_weight, cost_gradient_bias = result
                #     for j in range(len(self.weights)):
                #         cost_gradient_weights[j] += cost_gradient_weight[j]
                #         cost_gradient_biases[j]  += cost_gradient_bias[j]

                for input, target in mini_batch:
                    cost_gradient_weight, cost_gradient_bias = self.backpropagation(input, target)
                    for j in range(len(self.weights)):
                        cost_gradient_weights[j] += cost_gradient_weight[j]
                        cost_gradient_biases[j]  += cost_gradient_bias[j]


                for j in range(len(self.weights)):
                    self.weights[j] -= learning_rate * cost_gradient_weights[j]
                    self.biases[j]  -= learning_rate * cost_gradient_biases[j]

            if test_data:
                correct = 0
                mse = None
                for input, target in test_data:
                    result, _ = self.feed_forward(input)
                    result = result[-1]
                    
                    if type(mse).__name__ == "NoneType":
                        mse = np.zeros(result.T.shape)

                    mse += np.square(target - result.T)

                    if padding != 0:
                        correct += 0.5
                        # print(result, target)
                        # if result[0]-padding <= target[0] and result[0]+padding >= target[0]:
                            # correct += 1
                    elif np.argmax(result) == np.argmax(target):
                        correct += 1

                mse /= (len(test_data))
                mse = np.mean(mse)

                print(f"Loss: {mse}")
                y2.append(mse)

                accuracy = 100*correct/len(test_data)
                print(f"Accuracy: {accuracy}%")
                y.append(accuracy)
                new_time = time.time()
                print(f"Time: {round(new_time-prev_time, 2)} seconds")
                prev_time = new_time
                # print(abs(accuracy-prev_accuracy))
                if abs(accuracy - prev_accuracy) <= patience_threshold:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if accuracy == 100:
                    print("Perfection")
                    break

                if patience_counter > patience:
                    print("Network not improving fast enough, quit to avoid overfitting")
                    break

                prev_accuracy = accuracy

        f = open('network/weights.txt', 'w')
        f.write(str(self.weights))
        f.close()

        f = open('network/biases.txt', 'w')
        f.write(str(self.biases))
        f.close()

        y.insert(0, 0)
        plt.plot(list(range(0, len(y))), y)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy vs Epochs")
        plt.show()

        plt.plot(list(range(0,len(y2))), y2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs Epochs")
        plt.show()

def benchmark():
    input_values_list = [[0,0], [0,1], [1,0], [1,1]]
    target_values_list = [[1,0], [0,1], [0,1], [1,0]]

    new_input = [np.mat(x).T for x in input_values_list]
    new_target = [np.mat(x) for x in target_values_list]

    test_data = list(zip(new_input, new_target))

    times = 10
    input_values = [x for x in new_input for _ in range(times)]
    target_values = [x for x in new_target for _ in range(times)]

    neural_network = Network([2, 3, 2])
    neural_network.train(input_values, target_values, epochs=3000,  mini_batch_size=2000, learning_rate=0.8, patience=np.inf, test_data=test_data)

def make_linear_regression(sets=1, points=1000, scale=10):
    inputs = []
    outputs = []

    for _ in range(sets):
        noise1 = list(np.random.randn(1, points))

        m = np.random.randint(-100, 100)
        b = np.random.randint(-100, 100)

        pts = [[x, m * x + b + noise1[0][x] * scale] for x in range(points)]
        
        s_x = 0
        s_y = 0
        s_xx = 0
        s_xy = 0

        p = []

        for x, y in pts:
            s_x += x
            s_y += y
            s_xx += x*x
            s_xy += x*y
            p.append(x)
            p.append(y)
        
        m = (points*s_xy - s_x*s_y)/(points*s_xx - s_x*s_x)
        b = (s_y - m * s_x) / points
        
        input = np.asmatrix(p)
        output = np.asmatrix([m, b])

        inputs.append(input.T)
        outputs.append(output)

    return (inputs, outputs)


if __name__ == "__main__":
    rows = []
    with open('network/mnist_train.csv') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)

    target_values = [np.asmatrix([1 if i == int(row[0]) else 0 for i in range(10)]) for row in rows]
    input_values = [np.asmatrix(list(map(lambda x : x / 255, map(int, row[1:])))).T for row in rows]

    rows = []
    with open('network/mnist_test.csv') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)

    target_values_test = [np.asmatrix([1 if i == int(row[0]) else 0 for i in range(10)]) for row in rows]
    input_values_test = [np.asmatrix(list(map(lambda x : x / 255, map(int, row[1:])))).T for row in rows]

    test_data = list(zip(input_values_test, target_values_test))
    neural_network = Network([784, 32, 10])
    neural_network.train(input_values, target_values, epochs=50,  mini_batch_size=16, learning_rate=0.1, patience=5, test_data=test_data)


    # input_values, target_values = make_linear_regression(100, 500, 10)
    # input_values_test, target_values_test = make_linear_regression(100, 500, 5)

    # test_data = list(zip(input_values_test, target_values_test))

    # neural_network = Network([1000, 32, 2])
    # neural_network.train(input_values, target_values, epochs=50,  mini_batch_size=16, learning_rate=0.8, patience=5, padding=2, test_data=test_data)
    

    # benchmark()


    