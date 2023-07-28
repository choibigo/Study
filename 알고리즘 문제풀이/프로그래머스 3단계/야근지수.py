import heapq

def solution(n, works):
    works = list(map(lambda x : -x, works))
    heapq.heapify(works)
    
    while works and n:
        pop_num = heapq.heappop(works)
        
        if pop_num+1 != 0:
            heapq.heappush(works, pop_num+1)
        n-=1
    
    return sum(list(map(lambda x : x**2, works))) if len(works) else 0
