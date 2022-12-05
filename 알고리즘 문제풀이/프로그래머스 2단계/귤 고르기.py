from collections import Counter

def solution(k, tangerine):

    info = sorted(Counter(tangerine).items(), key=lambda x : -x[1])
    
    sum_value = 0
    for result, (key, count) in enumerate(info, 1):
        sum_value += count
        if sum_value>=k:
            return result