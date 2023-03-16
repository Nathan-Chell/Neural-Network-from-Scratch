#   
#   

import os
import csv
import math

import Neuron as n
import Activation_funcs as af

from random import random


def get_data():
    
    cur_path = os.getcwd()
    parent_path = os.path.dirname(cur_path)
    data_path = os.path.join(parent_path, "data")
    
    #   Training data
    with open(os.path.join(data_path, "mnist_train.csv")) as csv_file:
        reader = csv.reader(csv_file)
        csv_data = [row for row in reader]
        
        # Remove the header row from the CSV data
        csv_data = csv_data[1:]
        
        # Extract the features (pixel values) and labels from the CSV data
        X_train = [row[1:] for row in csv_data]
        y_train = [row[0] for row in csv_data]

    #   Testing data
    with open(os.path.join(data_path, "mnist_test.csv")) as csv_file:
        reader = csv.reader(csv_file)
        csv_data = [row for row in reader]
        
        # Remove the header row from the CSV data
        csv_data = csv_data[1:]
        
        # Extract the features (pixel values) and labels from the CSV data
        X_test = [row[1:] for row in csv_data]
        y_test = [row[0] for row in csv_data]
        
        
    return X_train, y_train, X_test, y_test

def CleanData(data):
    return [int(i) for i in data]

def NormilizeData(data):
    norm_arr = []
    
    for k in data:
        #   Convert each element in data into an integer and store in a new list
        int_data = CleanData(k)
        
        #   Find the maximum and minimum values in the original data
        max_value = max(int_data)
        min_value = min(int_data)
        
        #   Normalize each element in the integer data based on the max and min values
        #   The result will be a list of normalized values between 0 and 1
        norm_arr.append([(x - min_value) / (max_value - min_value)for x in int_data])
        
    return norm_arr

def ArithmeticMean(arr):
    return sum(arr) / len(arr)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):

        #   Initialize weights to be values from -1 to 1
        #   Initialize biases to be 0

        self.weights = [[(random() * 2 - 1) for i in range(n_inputs)]
                        for j in range(n_neurons)]
        self.biases = [0 for i in range(n_neurons)]

    def forward(self, inputs):
        self.output = n.matrixDotProduct(inputs, self.weights, self.biases)
        
class Loss:
    def Compute(self, output, target):
        
        sample_losses = self.forward(output, target)
        batch_loss = ArithmeticMean(sample_losses)
        
        return batch_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_target):

        correct_confidences = []
        negative_log = []
        samples = len(y_pred)
        
        #   Clip data to prevent division by 0
        y_pred_clipped = [[max(min(i, 1 - 1e-15), 1e-15) for i in k] for k in y_pred]
        
        #   Only accepting scalar values
        #correct_confidences = y_pred_clipped[range(samples), y_target]
        for target_element, pred_element in zip(y_target, y_pred_clipped):
            correct_confidences.append(pred_element[target_element])
        
        for i in range(samples):
            negative_log.append(-math.log(correct_confidences[i]))
            
        return negative_log

def main():
    
    #   Load in the MNIST dataset,
    #   It has already been split into training and testing data
    print("Loading data...")
    X_train, y_train, X_test, y_test = get_data()
    
    #   The data needs to be converted to a float and normalized
    print("Normalizing data...")
    X_train = NormilizeData(X_train)
    y_train = CleanData(y_train)
    X_test = NormilizeData(X_test)
    y_test = CleanData(y_test)
    
    print("Done!")

    Activation = af.ReLU()
    Activation2 = af.Softmax()
    
    Layer1 = Layer_Dense(784, 5)
    Layer2 = Layer_Dense(5, 10)
    
    Layer1.forward(X_train)
    Activation.forward(Layer1.output)
    
    Layer2.forward(Activation.output)
    Activation2.forward(Layer2.output)
    
    
    loss_func = Loss_CategoricalCrossEntropy()
    loss = loss_func.Compute(Activation2.output, y_train)
    print(loss)
    

if __name__ == "__main__":
    main()