#
#   Softmax activation function
#   f(x) = e^x / sum(e^x)
#
#   Learn more: https://en.wikipedia.org/wiki/Softmax_function

class Softmax:
    def __init__(self):
        #   Constant e
        #   This amount of precision is good for 15 decimal places
        self.e = 2.718281828459045

    def forward(self, x):
        self.output = [[(self.e**k) / sum([self.e**j for j in x[i]])
                        for k in x[i]] for i in range(len(x))]
