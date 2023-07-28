import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    weighted_sum = np.sum(w*x) + b

    if weighted_sum <= 0:
        return 0
    return 1

if __name__ == "__main__":
    print(AND(1,1)) # 1
    print(AND(1,0)) # 0
    print(AND(0,1)) # 0
    print(AND(0,0)) # 0