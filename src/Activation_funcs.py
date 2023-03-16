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


class Sigmoid:  
    def __init__(self):
        #   Constant e
        #   This amount of precision is good for 15 decimal places
        self.e = 2.718281828459045
    
    def forward(self, x):
        self.output = [[1 / (1 + self.e**(-k)) for k in j] for j in x]
    
class ReLU:
    def forward(self, x):
        self.output = [[max(0, k) for k in j] for j in x]

class tanh:  
    def __init__(self):
        #   Constant e
        #   This amount of precision is good for 15 decimal places
        self.e = 2.718281828459045

    def forward(self, x):
        self.output = [[((self.e**k - self.e**(-k)) / (self.e**k + self.e**(-k)))
                        for k in j] for j in x]
    

class Softmax:  
    def __init__(self):
        #   Constant e
        #   This amount of precision is good for 15 decimal places
        self.e = 2.718281828459045

    def forward(self, x):
        
        self.output = [[(self.e**k) / sum([self.e**j for j in x[i]])for k in x[i]] for i in range(len(x))]

