import heapq

def solution(scoville, K):

    count = 0
    heapq.heapify(scoville)
    while len(scoville) >= 2:
        s1 = heapq.heappop(scoville)
        s2 = heapq.heappop(scoville)
        
        if s1 >= K:
            return count
        
        heapq.heappush(scoville, (s1 + s2*2))
        count +=1
    
    if len(scoville) == 1 and scoville[0]>=K:
        return count
    
    return -1