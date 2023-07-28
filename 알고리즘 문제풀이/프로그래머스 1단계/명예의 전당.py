from heapq import heappush, heappop
from collections import deque

def solution(k, score):
    score = deque(score)
    result = list()
    heap = list()
    
    for pop_s in score:
        if len(heap) ==k:
            if heap[0] < pop_s:
                heappop(heap)
                heappush(heap, pop_s)
        else:
            heappush(heap, pop_s)
        result.append(heap[0])
    
    return result