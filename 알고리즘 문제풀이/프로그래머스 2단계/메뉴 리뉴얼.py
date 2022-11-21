from itertools import combinations
from collections import defaultdict

def solution(orders, course):
    
    for i in range(len(orders)):
        orders[i] = sorted(orders[i])
        
    answer = list()
    for c in course:
        info = defaultdict(int)
        for order in orders:
            for food in combinations(order, c):
                info["".join(food)] += 1

        if info:
            max_count = max(info.values())
            if max_count > 1:
                for key, value in info.items():
                    if value == max_count:
                        answer.append(key)
    
    return sorted(answer)
