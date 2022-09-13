import heapq

def solution(n, works):
    if sum(works) <= n : return 0

    works = list(map(lambda x : -x, works))
    heapq.heapify(works)
    
    for _ in range(n):
        temp = heapq.heappop(works)
        heapq.heappush(works, temp+1)
        
    result = 0
    for work in works:
        result += (work**2)
    
    return result