from itertools import permutations

def solution(k, dungeons):
    
    dungeons_count = len(dungeons)
    answer = 0
    
    for courses in list(permutations(dungeons, dungeons_count)):
        temp_k = k
        count = 0
        for need, consum in courses:
            if need <= temp_k:
                temp_k -= consum
                count +=1
                answer = max(answer, count)
            else:
                answer = max(answer, count)
                break
                
    return answer