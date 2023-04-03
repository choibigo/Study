import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a

if __name__ == "__main__":
    print(softmax(np.array([1010, 1000, 990]))) # [9.99954600e-01 4.53978686e-05 2.06106005e-09]
    