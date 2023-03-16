#   Implementation of common activation functions
#
#   Sigmoid (logistic) function
#   f(x) = 1 / (1 + e^-x)
#
#   Leanr more: https://en.wikipedia.org/wiki/Sigmoid_function
#
#   ReLU (Rectified Linear Unit) function
#   f(x) = max(0, x)
#
#   Leanr more: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)s
#
#   tanh function
#   f(x) = (e^x - e^-x) / (e^x + e^-x)
#
#   Leanr more: https://en.wikipedia.org/wiki/Hyperbolic_functions


def sigmoid(x):
    
    #   Constant e
    #   This amount of precision is good for 15 decimal places
    e = 2.718281828459045
   
    return (1 / (1 + e**(-x)))
    
def ReLU(x):
    
    return max(0, x)

def tanh(x):
    
    #   Using e from sigmoid function
    e = 2.718281828459045
    
    return ((e**x - e**(-x)) / (e**x + e**(-x)))

