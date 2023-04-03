import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    input_data = np.array([-0.5, 0.3, 0.7])
    print(sigmoid_function(input_data)) # [0.37754067 0.57444252 0.66818777]