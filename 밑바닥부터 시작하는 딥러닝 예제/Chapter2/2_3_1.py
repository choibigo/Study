def AND(x1, x2):
    w1 = 0.5
    w2 = 0.5
    theta = 0.7
    weighted_sum = w1*x1 + w2*x2

    if weighted_sum <= theta:
        return 0
    else:
        return 1
    
if __name__ == "__main__":
    print(AND(1,1)) # 1
    print(AND(1,0)) # 0
    print(AND(0,1)) # 0
    print(AND(0,0)) # 0