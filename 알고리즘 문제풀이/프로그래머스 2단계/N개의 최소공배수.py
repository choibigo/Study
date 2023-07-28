def func(num1, num2):
    origin_num1 = num1
    origin_num2 = num2
    
    while num2:
        num1, num2 = num2, num1%num2
    
    return int(origin_num1*origin_num2 / num1)

def solution(arr):
    
    num = arr[0]
    for a in arr[1:]:
        num = func(num, a)
        
    return num