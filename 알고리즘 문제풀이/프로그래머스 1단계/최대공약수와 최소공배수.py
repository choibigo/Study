def func1(num1, num2):
    while(num2):
        num1, num2 = num2, num1%num2
        
    return num1

def func2(num1, num2, tt):
    return (num1 * num2) / tt

def solution(n, m):
    answer = [0,0]
    
    answer[0] = func1(n, m)
    answer[1] = func2(n, m, answer[0])
    
    return answer