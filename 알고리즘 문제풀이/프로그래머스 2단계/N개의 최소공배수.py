from collections import deque

def func1(num1, num2):
    
    target1 = num1
    target2 = num2
    
    while num2:
        num1, num2 = num2, num1%num2
    
    return (target1*target2)/num1

def solution(arr):
    
    arr = deque(arr)
    
    while len(arr) > 1:
        pop_num1 = arr.popleft()
        pop_num2 = arr.popleft()
    
        temp = func1(pop_num1, pop_num2)
        arr.append(temp)
        
    return int(arr[-1])