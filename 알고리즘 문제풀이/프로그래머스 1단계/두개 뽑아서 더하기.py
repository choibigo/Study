from itertools import combinations

def solution(numbers):
    answer = set()
    
    for temp in list(combinations(numbers, 2)):
        answer.add(sum(temp))
    
    answer = list(answer)
    answer.sort()
    
    return answer