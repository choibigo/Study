def factorial(num):
    if num == 1:
        return 1
    
    return num * factorial(num-1)

def solution(n, k):
    
    answer = list()
    num_list = [x for x in range(1, n+1)]
    
    while n > 0:
        temp = factorial(n) // n
        
        index = k // temp
        k %= temp
        
        if k == 0:
            answer.append(num_list.pop(index-1))
        else:
            answer.append(num_list.pop(index))
            
        n-=1
            
    return answer