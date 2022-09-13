def solution(n, s):
    
    if n > s : return [-1]
    
    init = s // n
    result = [init for _ in range(n)]
    
    for i in range(1, s % n+1):
        result[-i] += 1

        
    return result