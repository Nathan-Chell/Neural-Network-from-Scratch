#
#   tanh function
#   f(x) = (e^x - e^-x) / (e^x + e^-x)
#
#   Learn more: https://en.wikipedia.org/wiki/Hyperbolic_functions


class tanh:
    def __init__(self):
        #   Constant e
        #   This amount of precision is good for 15 decimal places
        self.e = 2.718281828459045

    def forward(self, x):
        self.output = [[((self.e**k - self.e**(-k)) / (self.e**k + self.e**(-k)))
                        for k in j] for j in x]
