import numpy as np

class Network():
    def __init__(self):
        self.network = {}
        self.network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        self.network['b1'] = np.array([0.1, 0.2, 0.3])
        self.network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        self.network['b2'] = np.array([0.1, 0.2])
        self.network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.network['b3'] = np.array([0.1, 0.2])

    def __call__(self, x):
        return self.__forward__(x)
    
    def __forward__(self, x):
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = self.__sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.__sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        output = a3

        return output
    
    def __sigmoid(self, x):
        return 1 / (1+ np.exp(-x))

if __name__ == "__main__":

    input_x = np.array([1.0, 0.5])

    my_network = Network()
    output = my_network(input_x)

    print(output) # [0.31682708 0.69627909]
