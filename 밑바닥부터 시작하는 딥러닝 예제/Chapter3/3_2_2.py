import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int32)
if __name__ == "__main__":
    input_data = np.array([-0.5, 0.3, 0.7])
    print(step_function(input_data)) # [0 1 1]