import heapq

def solution(operations):

    min_heap = list()
    max_heap = list()
    
    for temp in operations:
        op, num = temp.split(" ")
        num = int(num)
        if op == "I":
            heapq.heappush(min_heap, num)
            heapq.heappush(max_heap, -num)
        elif op == "D" and len(min_heap):
            if num == 1:
                pop_num = heapq.heappop(max_heap)
                min_heap.remove(-pop_num)
            else:
                pop_num = heapq.heappop(min_heap)
                max_heap.remove(-pop_num)
                
    return [-heapq.heappop(max_heap), heapq.heappop(min_heap)] if len(min_heap) else [0,0]
        