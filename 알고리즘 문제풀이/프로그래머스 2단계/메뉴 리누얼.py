from itertools import combinations
from collections import Counter

def solution(orders, course):
    
    answer = list()
    
    for c in course:
        res = list()
        for order in orders:
            res += list(combinations(sorted(order), c))
            
        count = Counter(res)
        
        if count:
            if max(count.values()) >= 2:
                for key, value in count.items():
                    if value == max(count.values()):
                        answer.append("".join(key))
        
    return sorted(answer)