import heapq

def solution(operations):
    
    min_heap = []
    max_heap = []
    
    for operation in operations:
        op, num = operation.split(" ")
        num = int(num)
        
        if op == "I":
            heapq.heappush(min_heap, num)
            heapq.heappush(max_heap, num*-1)
    
        else:
            if len(min_heap) != 0:
                if num == -1:
                    temp = heapq.heappop(min_heap)
                    max_heap.remove(temp*-1)
                elif num == 1:
                    temp = heapq.heappop(max_heap)
                    min_heap.remove(temp*-1)
                    
    return [0,0] if len(min_heap) == 0 else [heapq.heappop(max_heap)*-1, heapq.heappop(min_heap)]