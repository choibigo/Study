def f(x):
    if x%2 ==0:
        return x+1
    else:
        y = "0" + bin(x)[2:]
        idx = y.rfind("0")
        y = list(y)
        y[idx] = "1"
        y[idx+1] = "0"
        
        return int(''.join(y), 2)
        

def solution(numbers):
                
    return [f(x) for x in numbers]
