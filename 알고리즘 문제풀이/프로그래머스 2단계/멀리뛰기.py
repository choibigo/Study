def solution(n):
    
    if n <= 2:
        return n
    
    res = [0] * (n+1)
    res[1] = 1
    res[2] = 2
    
    for i in range(3, n+1):
        res[i] = (res[i-1] + res[i-2]) % 1234567
        
    return res[n]