import math

def solution(n, stations, w):
    
    answer = 0
    index = 1
    
    for s in stations:
        answer += max(math.ceil((s-w-index)/ (w*2+1)), 0)
        index = s+w+1

    if index <= n:
        answer += max(math.ceil((n-index+1)/ (w*2+1)), 0)
    
        
    return answer