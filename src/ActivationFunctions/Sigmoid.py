#
#   Sigmoid (logistic) function
#   f(x) = 1 / (1 + e^-x)
#
#   Learn more: https://en.wikipedia.org/wiki/Sigmoid_function

class Sigmoid:
    def __init__(self):
        #   Constant e
        #   This amount of precision is good for 15 decimal places
        self.e = 2.718281828459045

    def forward(self, x):
        self.output = [[1 / (1 + self.e**(-k)) for k in j] for j in x]
