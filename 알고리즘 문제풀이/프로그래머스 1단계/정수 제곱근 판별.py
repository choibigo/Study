def solution(n):
    
    temp = n ** (1/2)
    if temp % 1 == 0:
        return (temp+1)**2
    
    else:
        return -1
    