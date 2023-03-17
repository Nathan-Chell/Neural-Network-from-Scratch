#
#   ReLU (Rectified Linear Unit) function
#   f(x) = max(0, x)
#
#   Learn more: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)s


class ReLU:
    def forward(self, x):
        self.output = [[max(0, k) for k in j] for j in x]
