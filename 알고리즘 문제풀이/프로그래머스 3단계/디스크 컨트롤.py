import heapq

def solution(jobs):

    start = -1
    now = 0
    heap = list()
    i=0
    answer= 0
    
    while i < len(jobs):
        for j in jobs:
            if start<j[0]<=now:
                heapq.heappush(heap, [j[1], j[0]])
                
        if len(heap):
            current = heapq.heappop(heap)
            start = now
            now += current[0]
            answer += (now - current[1])
            i+=1
        else:
            now +=1
            
    return int(answer / len(jobs))