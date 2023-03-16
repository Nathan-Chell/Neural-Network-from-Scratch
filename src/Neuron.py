#
#   A simple neuron, perceptron, that takes x inputs aand weights
#

import Activation_funcs as af


# Compute the dot product between the inputs and the weights, and add the bias
# Inputs: a list of input values
# Weights: a list of weight values, with the same length as inputs
# Bias: a scalar value
# Returns: the dot product plus the bias

def dotProduct(inputs, weights, bias):
    x = 0
    for i in range(len(inputs)):
        x += inputs[i] * weights[i]
    return x + bias


# Compute the dot product between each batch of inputs and each neuron's weights, and add the neuron's bias
# Inputs: a list of input batches, where each batch is a list of input values
# Weights: a list of weight matrices, where each matrix has the same number of columns as the length of each input batch, and the number of rows is the number of neurons in the layer
# Bias: a list of bias values, where each value corresponds to a neuron in the layer
# Returns: a list of output batches, where each output batch is a list of output values for each neuron

def matrixDotProduct(inputs, weights, bias):
    outputs = []
    for k in inputs:
        outputs.append([dotProduct(k, weights[i], bias[i]) for i in range(len(weights))])
    return outputs


# Compute the output of a single neuron, using an activation function
# Inputs: a list of input values
# Weights: a list of weight values, with the same length as inputs
# Bias: a scalar value
# ActivationFunction: a function that takes a scalar value as input and returns a scalar value
# Returns: the output of the neuron, after applying the activation function

def NeuronOutput(inputs, weights, bias, activationFunction):
    output = dotProduct(inputs, weights, bias)
    return activationFunction(output)

def formatPrint(matrix):
    for i in matrix:
        print(i)


def main():
    
    #   EXAMPLE USAGE
    
    
    #   4 inputs
    #   Batch size of 3
    inputs = [[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]
    
    #   2 layers of size 3 each
    #   4 inputs for the first, from the inputs
    weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
    
    biases = [2, 3, 0.5]
    
    #   3 inputs for the second layer, from the 3 neurons of the first layer
    weights2 = [[0.1, -0.14, 0.5],
               [-0.5, 0.12, -0.33],
               [-0.44, 0.73, -0.13]]

    biases2 = [-1, 2, -0.5]
    
    layer1_output = matrixDotProduct(inputs, weights, biases)
    layer2_output = matrixDotProduct(layer1_output, weights2, biases2)
    
    print(formatPrint(layer2_output))



if __name__ == "__main__":
    main()
    