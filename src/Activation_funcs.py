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


class sigmoid:  
    def __init__(self):
        #   Constant e
        #   This amount of precision is good for 15 decimal places
        e = 2.718281828459045
    
    def forward(self, x):
        self.output = 1 / (1 + e**(-x))
    
class ReLU:
    def forward(self, x):
        self.output = max(0, x)

class tanh:  
    def __init__(self):
        #   Constant e
        #   This amount of precision is good for 15 decimal places
        e = 2.718281828459045

    def forward(self, x):
        self.output = ((e**x - e**(-x)) / (e**x + e**(-x)))
    

