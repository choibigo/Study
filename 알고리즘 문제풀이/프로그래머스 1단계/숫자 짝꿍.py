from collections import Counter

def solution(X, Y):
    
    x_counter = dict(Counter(X))
    y_counter = dict(Counter(Y))

    temp = list()
    for x_key, x_value in x_counter.items():
        y_value = y_counter.get(x_key, 0)
        
        if x_value and y_value:
            temp = temp + [x_key]*min(int(y_value), int(x_value))
    
    if len(temp) == 0:
        return "-1"
    elif len(temp) == temp.count("0"):
        return "0"
    else:
        temp.sort(reverse=True)
        return "".join(temp)
    
