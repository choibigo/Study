from collections import deque

def solution(prices):
    
    prices = deque(prices)
    answer = list()
    
    while prices:
        
        pop_p = prices.popleft()
        
        sec = 0
        for p in prices:
            sec+=1
            if pop_p > p:
                break
        answer.append(sec)
        
    return answer