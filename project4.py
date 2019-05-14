"""
Project 4
Alec Zemborain

How to run: In main is an example of how I ran and tested my code. In order to conduct your own test,
you must simply assign the file variable in main to a string containing the name of the file in the
directory you would like to use. The necessary functions to get the data in the right format
are already called. You can then initialize a NeuralNetwork of the dimensions you desire.
This is done by calling nn = NeuralNetwork([x1,x2,....,xn]). The only restrictions on the
dimensions of the nn is that the first layer must have the same number of nodes as the number
of variables in the data set (besides the targets, of course). The number of output nodes
is arbitrary, however, you may have to come up with your own way to evaluate the outputs
if you get creative with this aspect.
"""

import csv, sys, random, math, copy

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def hard_threshold(x):
    """ returns 1 if x > 0, 0 otherwise"""
    return int(x > 0)

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.
    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.
    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

def activation(x):
    """ function that allows to change activation function used
    without having to change the rest of the code"""
    return logistic(x)

def helper(w, d, j):
    """equivalent to sum from 0 to j of wi,j*delta[j] from textbook.
    my choice of notation made back propogation a bit more difficult,
    prompting me to have to create this function"""
    result = 0
    for i in range(0,len(d)):
        result += d[i] * w[i][j]
    return result

class NeuralNetwork():


    def __init__(self, shape):
        """initialize nodes and weights"""
        self.network = []
        self.weights = [[]]

        for item in shape:
            layer = [0] * item
            self.network.append(layer)

        for i in range(1, len(self.network)):
            layer_w = []
            for j in range(0,len(self.network[i])):
                node_w = []
                for k in range(0,len(self.network[i-1])):
                    node_w.append(random.uniform(-2.5,2.5))
                layer_w.append(node_w)
            self.weights.append(layer_w)


    def forward_propagate(self, x):
        """At each layer, compute dot product of weights going into each node
        and all the nodes in the previous layer"""
        self.network[0] = x
        for i in range(1,len(self.network)):
            for j in range(0,len(self.network[i])):
                self.network[i][j] = activation(dot_product(self.weights[i][j], self.network[i-1]))
        return self.network


    def back_propagation_learning(self, examples):
        """initialize local variables"""
        deltas = copy.deepcopy(self.network)
        lr = 1

        """radomize weights"""
        for i in range(1, len(self.network)):
            for j in range(0,len(self.network[i])):
                for k in range(0,len(self.network[i-1])):
                    self.weights[i][j][k] = random.uniform(-2.5,2.5)

        for _ in range(0,100):

            """forward propagate examples"""
            for (example, target) in examples:
                self.network = self.forward_propagate(example)

                """Calculate deltas on output layer"""
                for i in range(0,len(deltas[-1])):
                    g_in = self.network[-1][i] * (1-self.network[-1][i])
                    deltas[-1][i] = g_in * (target[i] - self.network[-1][i])

                """back propagate deltas"""
                for i in range(len(self.network) - 2, 0, -1):
                    for j in range(0,len(self.network[i])):
                        g_in = self.network[i][j] * (1-self.network[i][j])
                        deltas[i][j] = g_in * helper(self.weights[i+1],deltas[i+1], j)

                """update weights"""
                for i in range(1, len(self.network)):
                    for j in range(0,len(self.network[i])):
                        for k in range(0,len(self.network[i-1])):
                            self.weights[i][j][k] = self.weights[i][j][k] + (lr * self.network[i-1][k] * deltas[i][j])


    def predict_class(self):
        """Used for single class prediction.
        Returns integer value of output node"""
        return int(self.network[-1][0] > 0.5)







def main():

    file = "breast-cancer-wisconsin-normalized.csv"

    file = read_data(file)
    file = convert_data_to_pairs(file[1],file[0])
    random.shuffle(file)

    training = file[0:int(len(file)*.6)]
    testing = file[int(len(file)*.6):]

    print("[2,2,1]")
    nn = NeuralNetwork([30,2,1])
    nn.back_propagation_learning(training)
    print(accuracy(nn, testing))

    print("[2,3,3,1]")
    nn = NeuralNetwork([30,3,3,1])
    nn.back_propagation_learning(training)
    print(accuracy(nn, testing))

    print("[2,4,4,4,1]")
    nn = NeuralNetwork([30,4,4,4,1])
    nn.back_propagation_learning(training)
    print(accuracy(nn, testing))



if __name__ == "__main__":
    main()
